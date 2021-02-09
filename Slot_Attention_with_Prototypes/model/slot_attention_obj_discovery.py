"""
Slot attention model based on code of tkipf and the corresponding paper Locatello et al. 2020
"""
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torchsummary import summary


def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    slots = torch.reshape(slots, [slots.shape[0] * slots.shape[1], 1, 1, slots.shape[2]])
    # `slots` has shape: [batch_size, num_slots, slot_size].
    grid = slots.repeat(1, resolution[0], resolution[1], 1)
    # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
    return grid


def unstack_and_split(x, batch_size, n_slots, num_channels=3):
  """Unstack batch dimension and split into channels and alpha mask."""
  # unstacked = torch.reshape(x, [batch_size, -1] + list(x.shape[1:]))
  # channels, masks = torch.split(unstacked, [num_channels, 1], dim=-1)
  unstacked = torch.reshape(x, [batch_size, n_slots] + list(x.shape[1:]))
  channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)
  return channels, masks


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.project_q = nn.Linear(dim, dim, bias=True)
        self.project_k = nn.Linear(dim, dim, bias=True)
        self.project_v = nn.Linear(dim, dim, bias=True)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_inputs = nn.LayerNorm(dim, eps=1e-05)
        self.norm_slots = nn.LayerNorm(dim, eps=1e-05)
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-05)

        self.attn = 0

    def forward(self, inputs, num_slots=None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_inputs(inputs)
        k, v = self.project_k(inputs), self.project_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.project_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_mlp(slots))

        self.attn = attn

        return slots


class SlotAttention_encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SlotAttention_encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class SlotAttention_decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SlotAttention_decoder, self).__init__()
        self.network = nn.Sequential(
            # nn.ZeroPad2d((-1, -1, -1, -1)),
            nn.ConvTranspose2d(in_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, (5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, (5, 5), stride=(1, 1), padding=2, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, 4, (3, 3), stride=(1, 1), padding=1),
        )

    def forward(self, x):
        return self.network(x)


class MLP(nn.Module):
    def __init__(self, hidden_channels):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x):
        return self.network(x)


class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        # self.grid = torch.FloatTensor(build_grid(resolution))
        # self.grid = self.grid.to(device)
        # for nn.DataParallel
        self.register_buffer("grid", torch.FloatTensor(build_grid(resolution)))
        self.resolution = resolution[0]
        self.hidden_size = hidden_size

    def forward(self, inputs):
        return inputs + self.dense(self.grid).view((-1, self.hidden_size, self.resolution, self.resolution))


class SlotAttention_classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SlotAttention_classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, in_channels),  # nn.Conv1d(in_channels, in_channels, 1, stride=1, groups=in_channels)
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class SlotAttention_model(nn.Module):
    def __init__(self, n_slots, n_iters, n_attr,
                 in_channels=3,
                 encoder_hidden_channels=64,
                 attention_hidden_channels=128,
                 decoder_hidden_channels=64,
                 decoder_initial_size=(8, 8)):
        super(SlotAttention_model, self).__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.n_attr = n_attr
        self.n_attr = n_attr + 1  # additional slot to indicate if it is a object or empty slot
        self.encoder_hidden_channels = encoder_hidden_channels
        self.decoder_hidden_channels = decoder_hidden_channels
        self.decoder_initial_size = decoder_initial_size

        self.encoder_cnn = SlotAttention_encoder(in_channels=in_channels, hidden_channels=encoder_hidden_channels)
        self.encoder_pos = SoftPositionEmbed(encoder_hidden_channels, (128, 128))
        self.layer_norm = nn.LayerNorm(encoder_hidden_channels, eps=1e-05)
        self.mlp = MLP(hidden_channels=encoder_hidden_channels)
        self.slot_attention = SlotAttention(num_slots=n_slots, dim=encoder_hidden_channels, iters=n_iters, eps=1e-8,
                                            hidden_dim=attention_hidden_channels)
        self.decoder_pos = SoftPositionEmbed(decoder_hidden_channels, decoder_initial_size)
        self.decoder_cnn = SlotAttention_decoder(in_channels=decoder_hidden_channels,
                                                 hidden_channels=decoder_hidden_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img, part):
        if part == 'enc':
            return self.forward_enc(img)
        elif part == 'dec':
            return self.forward_dec(img)
        else:
            return None
            
    def forward_enc(self, img):
        # `x` has shape: [batch_size, width, height, num_channels].
        # SLOT ATTENTION ENCODER
        self.img_shape = img.shape
        x = self.encoder_cnn(img)
        x = self.encoder_pos(x)
        x = torch.flatten(x, start_dim=2)
        # permute channel dimensions
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = self.mlp(x)
        slots = self.slot_attention(x)
        # slots has shape: [batch_size, num_slots, slot_size].

        return slots

    def forward_dec(self, slots):
        # SLOT ATTENTION DECODER
        x = spatial_broadcast(slots, self.decoder_initial_size)
        # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].

        # # permute back to pytorch dimension representation
        x = x.permute(0, 3, 1, 2)
        # # `x` has shape: [batch_size*num_slots, slot_size, width_init, height_init].
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x)
        # # `x` has shape: [batch_size*num_slots, num_channels+1, width, height].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = unstack_and_split(x, batch_size=slots.shape[0], n_slots=slots.shape[1])
        # `recons` has shape: [batch_size, num_slots, num_channels, width, height].
        # `masks` has shape: [batch_size, num_slots, 1, width, height].

        # Normalize alpha masks over slots.
        masks = self.softmax(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        # `recon_combined` has shape: [batch_size, num_channels, width, height].

        return recon_combined, recons, masks

if __name__ == "__main__":
    x = torch.rand(15, 3, 128, 128).cuda()
    net = SlotAttention_model(n_slots=11, n_iters=3, n_attr=18,
                              encoder_hidden_channels=64, attention_hidden_channels=128,
                              decoder_hidden_channels=64, decoder_initial_size=(8, 8))
    net = net.cuda()
    output = net(x)
    # print(output.shape)
    # summary(net, (3, 128, 128))


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import models.modules as modules
from models.torch_truncnorm.TruncatedNormal import TruncatedNormal
from torchvision.models import ViT_B_16_Weights
from .custom_vision_transformer import vit_b_16


class VT(nn.Module):
    def __init__(
        self,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        n_proto_vecs=[4, 2],
        lin_enc_size=256,
        proto_dim=1,
        softmax_temp=1.0,
        extra_mlp_dim=4,
        multiheads=False,
        train_protos=False,
        device="cpu",
    ):
        super(VT, self).__init__()

        self.n_proto_vecs = n_proto_vecs
        self.n_groups = len(n_proto_vecs)
        self.train_group = 0
        self.n_attrs = sum(n_proto_vecs)
        self.proto_dim = proto_dim
        self.lin_enc_size = lin_enc_size
        self.softmax_temp = [softmax_temp for _ in range(self.n_groups)]
        self.device = device
        self.multiheads = multiheads
        self.extra_mlp_dim = extra_mlp_dim
        self.train_protos = train_protos
        self.batch_size = 0  # dummy
        self.decoder_only = False

        # positions in one hot label vector that correspond to a class, e.g. indices 0 - 4 are for different colors
        self.attr_positions = list(np.cumsum(self.n_proto_vecs))
        self.attr_positions.insert(0, 0)

        self.vt_weights = ViT_B_16_Weights.DEFAULT
        self._vt_encoder = vit_b_16(weights=self.vt_weights)
        # Set model to eval mode
        self._vt_encoder.eval()

        self.preprocess = self.vt_weights.transforms()

        self._encoder = modules.Encoder(
            3, num_hiddens, num_residual_layers, num_residual_hiddens
        )
        self.latent_shape = tuple(self._encoder(torch.rand(1, 3, 64, 64)).shape[1:])
        self.latent_flat = np.prod(self.latent_shape)
        self.vt_flat = self._vt_encoder.hidden_dim

        self._decoder_proto = nn.ModuleList(
            [
                modules.Decoder(
                    num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens
                )
                for i in range(self.n_groups)
            ]
        )

        self._vt_linear = nn.Sequential(
            nn.Linear(self.vt_flat, self.lin_enc_size),
            nn.BatchNorm1d(self.lin_enc_size),
            nn.ReLU(),
        )
        self.extra_dims = (
            self.n_groups * self.extra_mlp_dim
            if self.multiheads
            else self.extra_mlp_dim
        )
        self._decoder_linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        sum(n_proto_vecs[: i + 1]) + (i + 1) * self.extra_mlp_dim,
                        self.latent_flat,
                    ),
                    nn.BatchNorm1d(self.latent_flat),
                    nn.ReLU(),
                )
                for i in range(self.n_groups)
            ]
        )

        self._discriminator = nn.Sequential(
            nn.Linear(self.vt_flat + self.n_attrs, 512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.Mish(),
            nn.Linear(512, 1),
        )

        if not self.multiheads:
            # LeakyRelu as in MarioNette
            self.split_mlp = nn.Sequential(
                nn.Linear(self.self.vt_flat, self.n_groups * self.proto_dim),
                nn.LeakyReLU(),
            )
        elif self.multiheads:
            # Alternative:
            # each group should have its own mlp head
            # tanh because prototype embeddings lie between -1 and 1
            self.split_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.vt_flat, self.proto_dim),
                        nn.LeakyReLU(),
                    )
                    for i in range(self.n_groups)
                ]
            )

        if self.multiheads:
            self.extra_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.vt_flat, 128),
                        nn.ReLU(),
                        nn.Linear(128, self.extra_mlp_dim),
                        nn.Sigmoid(),
                    )
                    for i in range(self.n_groups)
                ]
            )
        else:
            self.extra_mlp = nn.Sequential(
                nn.Linear(self.vt_flat, 128),
                nn.ReLU(),
                nn.Linear(128, self.extra_mlp_dim),
                nn.Sigmoid(),
            )

        self.proto_layer = self.init_prototypes()

        self.softmax = nn.Softmax(dim=1)
        # Separate n_groups channels into n_groups (equivalent with InstanceNorm)
        self.group_norm = nn.GroupNorm(self.n_groups, self.n_groups, eps=1e-05)

    def init_prototypes(self):
        self.proto_dict = dict()
        for group_id, num_protos in enumerate(self.n_proto_vecs):
            self.proto_dict[group_id] = nn.Embedding(num_protos, self.proto_dim).to(
                self.device
            )

            # truncated normal sampling
            tn_dist = TruncatedNormal(loc=0.0, scale=0.5, a=-1.0, b=1.0)
            self.proto_dict[group_id].weight.data = tn_dist.rsample(
                [num_protos, self.proto_dim]
            ).to(self.device)

            # protottypes are not learnable
            self.proto_dict[group_id].weight.requires_grad = self.train_protos

            if self.train_protos:
                self.register_parameter(
                    f"proto_dict:{group_id}:weight", self.proto_dict[group_id].weight
                )

    def set_train_group(self, group_id):
        assert group_id >= 0 and group_id < self.n_groups
        self.train_group = group_id
        # freeze and unfreeze relevant layers
        for id in range(self.n_groups):
            grad_bool = id == group_id
            # (un)freeze
            if self.multiheads:
                for param in self.split_mlps[id][0].parameters():
                    param.requires_grad = grad_bool
                for param in self.extra_mlps[id][0].parameters():
                    param.requires_grad = grad_bool

            self.proto_dict[id].weight.requires_grad = self.train_protos and grad_bool
        print(f"Switched to train group {group_id}")

    def freeze_all_train_groups(self, unfreeze=False):
        self.decoder_only = True
        # freeze and unfreeze relevant layers
        for id in range(self.n_groups):
            #  freeze
            if self.multiheads:
                for param in self.split_mlps[id][0].parameters():
                    param.requires_grad = unfreeze
                for param in self.extra_mlps[id][0].parameters():
                    param.requires_grad = unfreeze

            self.proto_dict[id].weight.requires_grad = self.train_protos and unfreeze
        print(f"Froze all train groups.")

    # ------------------------------------------------------------------------------------------------ #
    # fcts for forward passing

    def forward(self, imgs, shared_masks):
        (x0, x1) = imgs  # sample, negative sample

        self.batch_size = x0.shape[0]

        (x0, x1) = (self.preprocess(x0), self.preprocess(x1))

        # x: [B, 3, 64, 64] --> [B, F, W, H] --> [B, D]
        z0 = self.encode(x0)
        z1 = self.encode(x1)

        # this extracts the variation information, i.e. color jitter that is requires to properly reconstruct
        z0_extra = self.extra_mlp(z0)
        z1_extra = self.extra_mlp(z1)

        # [B, D] --> [B, G, D_P], D_P = D/G
        z0 = self.split(z0)
        z1 = self.split(z1)

        # compute distance to prototype embeddings and return softmin distance as label prediction
        (z0_proto, z1_proto), (
            pred0,
            pred1,
            pred0_swap,
            pred1_swap,
        ) = self._comp_proto_dists(z0, z1, shared_masks)

        if self.extra_mlp_dim != 0.0:
            pred0 = torch.cat((pred0, z0_extra), dim=1)
            pred1 = torch.cat((pred1, z1_extra), dim=1)
            pred0_swap = torch.cat((pred0_swap, z0_extra), dim=1)
            pred1_swap = torch.cat((pred1_swap, z1_extra), dim=1)

        # convert codes and extra encoding via decoder to reconstruction
        z0_proto_recon = self.proto_decode(pred0)
        z1_proto_recon = self.proto_decode(pred1)
        z0_proto_recon_swap = self.proto_decode(pred0_swap)
        z1_proto_recon_swap = self.proto_decode(pred1_swap)

        # remove the continuous variables from the final prediction
        if self.extra_mlp_dim != 0.0:
            pred0 = pred0[:, : -self.extra_mlp_dim]
            pred1 = pred1[:, : -self.extra_mlp_dim]

        return (pred0, pred1), (
            z0_proto_recon,
            z1_proto_recon,
            z0_proto_recon_swap,
            z1_proto_recon_swap,
        )

    def forward_single(self, imgs, discriminator=False, group_id=0):
        self.batch_size = imgs.shape[0]

        # x: [B, 3, 64, 64] --> [B, F, W, H] --> [B, D]
        imgs = self.preprocess(imgs)

        vt_z = self.encode(imgs)
        # z_flat = torch.clone(z)

        if self.multiheads:
            z_extra = (
                torch.stack(
                    [self.extra_mlps[i].forward(vt_z) for i in range(self.n_groups)]
                )
                .permute(1, 0, 2)
                .reshape((vt_z.shape[0], -1))
            )
        else:
            z_extra = self.extra_mlp(vt_z)

        # [B, D] --> [B, G, D_P], D_P = D/G
        z = self.split(vt_z)

        z_proto, preds = self._comp_proto_dists_single(z)

        if discriminator:
            # shuffle preds for discriminator
            z_flat = torch.flatten(vt_z, start_dim=1)
            real_data = torch.cat((preds, z_flat), dim=1)
            fake_data, max_vals = self.get_fake_data(group_id, z_flat, preds)
            disc_real_pred = self._discriminator(real_data)
            disc_fake_pred = self._discriminator(fake_data)

        if self.extra_mlp_dim != 0.0:
            preds = torch.cat((preds, z_extra), dim=1)

        if discriminator:
            # z_proto_recon = self.proto_decode(preds.detach().clone())
            z_proto_recon = self.proto_decode_multiple(
                preds.detach().clone(), group_id=group_id
            )
        else:
            z_proto_recon = self.proto_decode_multiple(preds, group_id=group_id)

        # remove the continuous variables from the final prediction
        if self.extra_mlp_dim != 0.0:
            preds = preds[:, : -self.extra_dims]

        if discriminator:
            return preds, z_proto_recon, disc_real_pred, disc_fake_pred, max_vals
        else:
            return preds, z_proto_recon

    # ------------------------------------------------------------------------------------------------ #
    # helper fcts for forward pass

    def get_disc_data(self, group_id, z, preds, threshold=0.5, step=0.01):
        if threshold <= 0:
            return torch.cat(
                [
                    torch.cat((torch.roll(preds, i, dims=0), z), dim=1)
                    for i in range(1, preds.shape[0])
                ],
                dim=0,
            )

        # get preds with high confidence
        low_index = sum(self.n_proto_vecs[:group_id])
        high_index = low_index + self.n_proto_vecs[group_id]

        # get preds with high confidence
        high_conf_mask = preds > threshold
        # remove preds of upcoming groups
        high_conf_mask[:, high_index:] = False

        if torch.all(~high_conf_mask):
            return self.get_disc_data(
                group_id, z, preds, threshold=threshold - step, step=step
            )

        max_vals = torch.max(preds[:, :high_index], dim=0)[0]

        all_concats = torch.cat([z, preds], dim=1)
        expanded_concats = all_concats.unsqueeze(0).expand(
            (all_concats.shape[0], -1, -1)
        )
        repeated_preds = torch.repeat_interleave(preds, preds.shape[0], dim=0).reshape(
            (preds.shape[0], preds.shape[0], -1)
        )

        all_concats = torch.cat([repeated_preds, expanded_concats], dim=2)

        matching_protos = all_concats[
            torch.all(
                (all_concats[:, :, :high_index] > threshold)
                == (
                    all_concats[:, :, -preds.shape[-1] : -preds.shape[-1] + high_index]
                    > threshold
                ),
                axis=2,
            )
            & torch.any((all_concats[:, :, :high_index] > threshold), axis=2)
            & torch.any(
                (
                    all_concats[:, :, -preds.shape[-1] : -preds.shape[-1] + high_index]
                    > threshold
                ),
                axis=2,
            )
        ]
        opposing_protos = all_concats[
            # clear proto unequal
            (
                ~torch.all(
                    (all_concats[:, :, :high_index] > threshold)
                    == (
                        all_concats[
                            :, :, -preds.shape[-1] : -preds.shape[-1] + high_index
                        ]
                        > threshold
                    ),
                    axis=2,
                )
                # proto unclear at all
                | torch.all((all_concats[:, :, :high_index] < threshold), axis=2)
            )
            # filter out exact same proto
            & torch.any(
                (
                    all_concats[:, :, :high_index]
                    != all_concats[
                        :, :, -preds.shape[-1] : -preds.shape[-1] + high_index
                    ]
                ),
                axis=2,
            )
        ]

        # lastly remove the other preds
        return (
            matching_protos[:, : -preds.shape[-1]],
            opposing_protos[:, : -preds.shape[-1]],
            max_vals,
        )

    def get_disc_data2(self, group_id, z, preds, threshold=0.5, step=0.01):
        if threshold <= 0:
            return torch.cat(
                [
                    torch.cat((torch.roll(preds, i, dims=0), z), dim=1)
                    for i in range(1, preds.shape[0])
                ],
                dim=0,
            )

        # get preds with high confidence
        low_index = sum(self.n_proto_vecs[:group_id])
        high_index = low_index + self.n_proto_vecs[group_id]

        # get preds with high confidence
        high_conf_mask = preds > threshold
        # remove preds of upcoming groups
        high_conf_mask[:, high_index:] = False

        if torch.all(~high_conf_mask):
            return self.get_disc_data(
                group_id, z, preds, threshold=threshold - step, step=step
            )

        matching_protos = []
        opposing_protos = []

        for i, pred in enumerate(preds):
            conf_mask = high_conf_mask[i]

            matching_mask = torch.zeros((high_conf_mask.shape[0]), dtype=bool)
            opposing_mask = torch.ones((high_conf_mask.shape[0]), dtype=bool)

            # include preds with certain equal prototype but don't include if conf_mask is unclear
            matching_mask[
                torch.all(high_conf_mask == conf_mask, axis=1)
                & torch.any(high_conf_mask, axis=1)
            ] = True
            # exclude preds with certain equal prototype but include unclear ones
            opposing_mask[
                torch.all(high_conf_mask == conf_mask, axis=1)
                & torch.any(high_conf_mask, axis=1)
            ] = False

            matching_z = z[matching_mask]
            opposing_z = z[opposing_mask]

            matching_protos.append(
                torch.cat(
                    [
                        torch.unsqueeze(pred, dim=0).expand(len(matching_z), -1),
                        matching_z,
                    ],
                    dim=1,
                )
            )
            opposing_protos.append(
                torch.cat(
                    [
                        torch.unsqueeze(pred, dim=0).expand(len(opposing_z), -1),
                        opposing_z,
                    ],
                    dim=1,
                )
            )

        matching_protos = torch.cat(matching_protos, dim=0)
        opposing_protos = torch.cat(opposing_protos, dim=0)

        return matching_protos, opposing_protos

    def get_disc_data3(self, group_id, z, preds, threshold=0.5, min_length=0, step=0.01):
        if threshold <= 0:
            return torch.cat(
                [
                    torch.cat((torch.roll(preds, i, dims=0), z), dim=1)
                    for i in range(1, preds.shape[0])
                ],
                dim=0,
            )

        # get preds with high confidence
        low_index = sum(self.n_proto_vecs[:group_id])
        high_index = low_index + self.n_proto_vecs[group_id]

        # get preds with high confidence
        high_conf_mask = preds > threshold
        # remove preds of upcoming groups
        high_conf_mask[:, high_index:] = False

        if torch.all(~high_conf_mask):
            return self.get_fake_data(
                group_id,
                z,
                preds,
                threshold=threshold - step,
                min_length=min_length,
                step=step,
            )

        fake_data = []

        for i, pred in enumerate(preds):
            conf_mask = high_conf_mask[i]

            mask = torch.ones((high_conf_mask.shape[0]), dtype=bool)
            # exclude pred itself
            mask[i] = False
            # exclude preds with certain equal prototype but include unclear ones
            mask[
                torch.all(high_conf_mask == conf_mask, axis=1)
                & torch.any(high_conf_mask, axis=1)
            ] = False

            opposing_z = z[mask]

            fake_data.append(
                torch.cat(
                    [
                        torch.unsqueeze(pred, dim=0).expand(len(opposing_z), -1),
                        opposing_z,
                    ],
                    dim=1,
                )
            )

        fake_data = torch.cat(fake_data, dim=0)

        return fake_data

    def get_fake_data(self, group_id, z, preds):
        low_index = sum(self.n_proto_vecs[:group_id])
        high_index = low_index + self.n_proto_vecs[group_id]
        max_vals = torch.max(preds[:, :high_index], dim=0)[0]
        return (
            torch.cat(
                [
                    torch.cat((torch.roll(preds, i, dims=0), z), dim=1)
                    for i in range(1, preds.shape[0])
                ],
                dim=0,
            ),
            max_vals,
        )

    def get_prototype_embeddings(self, group_id):
        proto_embeddings = self.proto_dict[group_id].weight
        # proto_embeddings = self.proto_layer_norm[group_id](proto_embeddings.permute(1, 0)).permute(1, 0)
        return proto_embeddings

    def split(self, z):
        if self.multiheads:
            z = torch.stack(
                [self.split_mlps[i].forward(z) for i in range(self.n_groups)]
            ).permute(1, 0, 2)
        else:
            z = self.split_mlp(z).reshape(-1, self.n_groups, self.proto_dim)
        # z = self.layer_norm(z.permute(0, 2, 1)).permute(0, 2, 1)
        # z = self.instance_norm(z)
        z = self.group_norm(z)
        return z

    def encode(self, x):
        with torch.no_grad():
            z_vt = self._vt_encoder(x)  # [B, F, W, H]
        return z_vt

    def proto_decode(self, preds):
        # reconstruct z
        z_proto = self._decoder_linear(preds)

        z_proto_recon = self._decoder_proto(
            z_proto.view(
                -1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]
            )
        )  # [B, 3, W, H]
        return z_proto_recon

    def proto_decode_multiple(self, preds, group_id):
        z_proto_recon = []
        for i in range(group_id + 1):
            idx = list(range(sum(self.n_proto_vecs[: (i + 1)]))) + list(
                range(
                    sum(self.n_proto_vecs),
                    sum(self.n_proto_vecs) + (i + 1) * self.extra_mlp_dim,
                )
            )
            pred = preds[:, idx]

            if i == group_id and i > 0:
                # set unnecessary preds to 0
                masked_pred = pred.clone()
                masked_pred[
                    :, sum(self.n_proto_vecs[:(i)]) : sum(self.n_proto_vecs[: (i + 1)])
                ] = 0
                masked_pred[
                    :, sum(self.n_proto_vecs[: (i + 1)]) + (i) * self.extra_mlp_dim :
                ] = 0

                masked_proto = self._decoder_linears[i](masked_pred)

                z_proto_recon += [
                    self._decoder_proto[i](
                        masked_proto.view(
                            -1,
                            self.latent_shape[0],
                            self.latent_shape[1],
                            self.latent_shape[2],
                        )
                    )
                ]  # [B, 3, W, H]

            # reconstruct z
            z_proto = self._decoder_linears[i](pred)

            z_proto_recon += [
                self._decoder_proto[i](
                    z_proto.view(
                        -1,
                        self.latent_shape[0],
                        self.latent_shape[1],
                        self.latent_shape[2],
                    )
                )
            ]  # [B, 3, W, H]

        return z_proto_recon

    # ------------------------------------------------------------------------------------------------ #
    # fcts for getting proto-code

    def _comp_proto_dists(self, inputs0, inputs1, shared_masks):
        """
        Computes the distance between each encoding to the prototype vectors and creates a latent prototype vector for
        each image.
        :param inputs0:
        :param inputs1:
        :param shared_masks:
        :return:
        """

        # shared_masks: [G, B]
        # [B, G, L] --> [G, B, L]
        inputs0 = inputs0.permute(1, 0, 2).contiguous()
        inputs1 = inputs1.permute(1, 0, 2).contiguous()

        # [B, G] --> [G, B]
        shared_masks = shared_masks.permute(1, 0)

        z0_protos = torch.clone(inputs0)
        z1_protos = torch.clone(inputs1)

        distances0_emb = torch.tensor([], device=self.device)
        distances0_emb_swap = torch.tensor([], device=self.device)
        distances1_emb = torch.tensor([], device=self.device)
        distances1_emb_swap = torch.tensor([], device=self.device)

        for group_id in range(self.n_groups):
            # get input of one group, Dims: [B, L]
            input0 = inputs0[group_id]
            input1 = inputs1[group_id]

            # Dims: [N_Emb, Emb_dim]
            proto_embeddings = self.get_prototype_embeddings(group_id)

            distances0 = self.softmax_dot_product(
                input0, proto_embeddings, group_id=group_id
            )
            distances1 = self.softmax_dot_product(
                input1, proto_embeddings, group_id=group_id
            )
            # print(np.round(distances0[0].detach().cpu().numpy(), 2))

            # if the group id is
            if group_id < shared_masks.shape[0]:
                # shared_labels: 0 means the attributes should not be shared, otherwise it should
                bool_share = shared_masks[group_id].squeeze()

                # swap the distances for those attributes to be shared
                distances0_swap = torch.clone(distances0)
                distances1_swap = torch.clone(distances1)
                distances0_swap[bool_share] = distances1[bool_share]
                distances1_swap[bool_share] = distances0[bool_share]
            else:
                # swap the distances for those attributes to be shared
                distances0_swap = torch.clone(distances0)
                distances1_swap = torch.clone(distances1)

            distances0_emb = torch.cat((distances0_emb, distances0), dim=1)
            distances1_emb = torch.cat((distances1_emb, distances1), dim=1)
            distances0_emb_swap = torch.cat(
                (distances0_emb_swap, distances0_swap), dim=1
            )
            distances1_emb_swap = torch.cat(
                (distances1_emb_swap, distances1_swap), dim=1
            )

            # get prototype indices for each image
            encoding_indices0 = torch.argmax(distances0, dim=1).unsqueeze(dim=1)
            encoding_indices1 = torch.argmax(distances1, dim=1).unsqueeze(dim=1)

            quantized0 = proto_embeddings[encoding_indices0].squeeze(dim=1)
            quantized1 = proto_embeddings[encoding_indices1].squeeze(dim=1)

            z0_protos[group_id, :, :] = quantized0
            z1_protos[group_id, :] = quantized1

            # # measure distance between chosen prototypes and original encoding (input0 and input1)
            # e_latent_loss0 = catch_nan(F.mse_loss(quantized0.detach(), input0))
            # e_latent_loss1 = catch_nan(F.mse_loss(quantized1.detach(), input1))
            #
            # vq_loss += (e_latent_loss0 + e_latent_loss1)/2

        z0_protos = z0_protos.permute(1, 0, 2)
        z1_protos = z1_protos.permute(1, 0, 2)

        return (z0_protos, z1_protos), (
            distances0_emb,
            distances1_emb,
            distances0_emb_swap,
            distances1_emb_swap,
        )

    def _comp_proto_dists_single(self, inputs):
        """
        Computes the distance between each encoding to the prototype vectors and creates a latent prototype vector for
        each image.
        :param inputs0:
        :param inputs1:
        :param shared_masks:
        :return:
        """

        # shared_masks: [G, B]
        # [B, G, L] --> [G, B, L]
        inputs = inputs.permute(1, 0, 2).contiguous()

        z_protos = torch.clone(inputs)

        distances_emb = torch.tensor([], device=self.device)

        for group_id in range(self.n_groups):
            if group_id <= self.train_group:
                # get input of one group, Dims: [B, L]
                input = inputs[group_id]

                # Dims: [N_Emb, Emb_dim]
                proto_embeddings = self.proto_dict[group_id].weight

                distances = self.softmax_dot_product(
                    input, proto_embeddings, group_id=group_id
                )

                distances_emb = torch.cat((distances_emb, distances), dim=1)

                # get prototype indices for each image
                encoding_indices = torch.argmax(distances, dim=1).unsqueeze(dim=1)

                quantized = proto_embeddings[encoding_indices].squeeze(dim=1)

                z_protos[group_id, :, :] = quantized
            else:
                distances = torch.full(
                    (inputs.shape[1], self.n_proto_vecs[group_id]),
                    1 / self.n_proto_vecs[group_id],
                    device=distances_emb.device,
                )
                distances_emb = torch.cat((distances_emb, distances), dim=1)
        z_protos = z_protos.permute(1, 0, 2)

        return z_protos, distances_emb

    def softmax_dot_product(self, enc, protos, group_id=0):
        """
        softmax product as in MarioNette (Smirnov et al. 2021)
        :param enc:
        :param protos:
        :return:
        """
        norm_factor = torch.sum(
            torch.cat(
                [
                    torch.exp(
                        torch.sum(enc * protos[i], dim=1) / np.sqrt(enc.shape[-1])
                    ).unsqueeze(dim=1)
                    for i in range(protos.shape[0])
                ],
                dim=1,
            ),
            dim=1,
        )
        sim_scores = torch.cat(
            [
                (
                    torch.exp(
                        torch.sum(enc * protos[i], dim=1) / np.sqrt(enc.shape[-1])
                    )
                    / norm_factor
                ).unsqueeze(dim=1)
                for i in range(protos.shape[0])
            ],
            dim=1,
        )

        # apply extra softmax to possibly enforce one-hot encoding
        sim_scores = self.softmax((1.0 / self.softmax_temp[group_id]) * sim_scores)

        return sim_scores

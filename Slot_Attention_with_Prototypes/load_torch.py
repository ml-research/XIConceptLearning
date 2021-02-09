import torch
from model.slot_attention_obj_discovery import SlotAttention_model

# load SlotAttention_model weigths
model = SlotAttention_model(n_slots=11, n_iters=3, n_attr=18).to('cuda:0')
model = torch.nn.DataParallel(model, device_ids=[0,1])
log = torch.load('logs/slot-attention-clevr-objdiscovery-14')
weights = log["weights"]
model.load_state_dict(weights, strict=True)

# freeze
for child in model.children():
    for param in child.parameters():
        param.requires_grad = False

# override forward in slot_attention_obj_discovery.SlotAttention_model or extract nn.Module's from the model to reuse in own architecture
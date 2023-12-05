import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin

class ProjectionModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cc_projection = torch.nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.cc_projection(x)

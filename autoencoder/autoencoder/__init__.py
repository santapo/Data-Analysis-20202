from collections import OrderedDict

import torch

import torchvision.models as models
import torch.nn as nn

from .vgg16_autoencoder import AutoEncoder, EncoderMerged

_extractor_factory = {
    'resnet18':     models.resnet18,
    'resnet34':     models.resnet34,
    'resnet50':     models.resnet50,
    'vgg16_bn':     EncoderMerged
}

_autoencoder_factory = {
    'vgg16_bn':     AutoEncoder
}

def get_extractor(model_name: str, pretrained: bool = True, checkpoint: str = None):
    try:
        get_backbone = _extractor_factory[model_name]
    except:
        print(f'This model name {model_name} is not defined. Feature extractor will be None')
        return
    if 'resnet' in model_name:
        backbone = get_backbone(pretrained=pretrained)
        modules = list(backbone.children())[:-2]
        model = nn.Sequential(*modules)
    if 'vgg16_bn' in model_name and checkpoint is not None:
        model = get_backbone(pretrained=pretrained)
        autoencoder_state_dict = torch.load(checkpoint)
        encoder_state_dict = autoencoder_state_dict.copy()
        encoder_state_dict['state_dict'] = OrderedDict((k[14:] if 'model.encoder.' in k else k,v) for k,v in encoder_state_dict['state_dict'].items())
        model.load_state_dict(torch.load(checkpoint), strict=False)
    return model

def get_autoencoder(model_name: str, pretrained: bool = True):
    try:
        get_autoencoder = _autoencoder_factory[model_name]
    except:
        print(f'This model name {model_name} is not defined.')
        return
    autoencoder = get_autoencoder(pretrained=pretrained)
    return autoencoder

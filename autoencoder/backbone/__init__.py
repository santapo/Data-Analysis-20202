import torch

from .vgg16_bn import AutoEncoder
from .dummy_model import DummyAE

_autoencoder_factory = {
    'dummy':        DummyAE,
    'vgg16_bn':     AutoEncoder
}


def get_autoencoder(model_name: str, pretrained: bool = True):
    try:
        get_autoencoder = _autoencoder_factory[model_name]
    except:
        print(f'This model name {model_name} is not defined.')
        return
    autoencoder = get_autoencoder(pretrained=pretrained)
    return autoencoder

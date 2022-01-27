from typing import Dict
from .CGAN import CGAN
from .GAN import GAN

def get_model(config:Dict):
    if config['training']['model'] == 'GAN':
        return GAN(config)
    elif config['training']['model'] == 'CGAN':
        return CGAN(config)
    else:
        raise Exception('Model not found.')
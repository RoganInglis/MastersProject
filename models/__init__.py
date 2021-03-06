from models.base_model import BaseModel
from models.basic_model import BasicModel
from models.reinforce_model import ReinforceModel
from models.reinforce_model_black_box import ReinforceModelBlackBox
from models.reinforce_model_black_box_reader import ReinforceModelBlackBoxReader

__all__ = [
    "BaseModel",
    "BasicModel",
    "ReinforceModel",
    "ReinforceModelBlackBox",
    "ReinforceModelBlackBoxReader"
]


def make_model(config):
    if config['model_name'] in __all__:
        return globals()[config['model_name']](config)
    else:
        raise Exception('The model name %s does not exist' % config['model_name'])


def get_model_class(config):
    if config['model_name'] in __all__:
        return globals()[config['model_name']]
    else:
        raise Exception('The model name %s does not exist' % config['model_name'])

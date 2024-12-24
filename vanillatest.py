'''
Main module
'''
import json
from model import model_van
from trainer import trainer
import torch
#from loss import losses
from torchinfo import summary

def main(_model:torch.nn.Module,_config:dict)->None:
    """set's the parameter and trains the model

    Args:
        _model (torch.nn.Module): model used
        _config (dict): model configuration
    """
    _model = _model.to(_config["model_configuration"]["device"])
    train = trainer.Trainer(_config,_model,torch.nn.L1Loss(),
                            log=False,project_name="alphaV2",
                            resume=False,seed=100)
    train.train()
    
if __name__ == '__main__':
    NUM_CHANNELS = 13
    HEIGHT = 128
    WIDTH = 128
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    with open('config.json','r',encoding="UTF8") as fl:
        config = json.load(fl)
    model = model_van.VanillaPred()
    print(summary(model,(16,13,128,128)))
    main(model,config)

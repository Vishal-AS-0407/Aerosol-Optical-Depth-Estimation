'''
Main module
'''
import json
from model import Vgglstm
from trainer import trainer
import torch

def main(_model:torch.nn.Module,_config:dict)->None:
    """set's the parameter and trains the model

    Args:
        _model (torch.nn.Module): model used
        _config (dict): model configuration
    """
    _model = _model.to(_config["model_configuration"]["device"])
    train = trainer.Trainer(_config,_model,torch.nn.MSELoss(),
                            log=True,project_name="alphaV1",
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
    model = Vgglstm.VGGLSTM(lstm_hidden_size=LSTM_HIDDEN_SIZE,
                            lstm_num_layers=LSTM_NUM_LAYERS,
                            num_channels=NUM_CHANNELS, image_size=HEIGHT)
    main(model,config)

import os
import torch
from .resnet import ResNet
from typing import Tuple, Union, Optional
from tqdm.auto import tqdm
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_model, save_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
try:
    from torch.optim.lr_scheduler import LRScheduler
except:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

class ResNetPredictor(nn.Module):
    mode = {"regression", "single_label_classification", "multi_label_classification"}
    def __init__(self,
                 resnet: ResNet,
                 optimizer: Optional[optim.Optimizer] = None,
                 lr_scheduler: Optional[LRScheduler] = None,
                 mode: str = "single_label_classification",
                 num_classes: int = 0,
                 dropout: float = 0.5,
                 optimize_predictor: bool = True):
        # nn.Dropout(p = 0) == nn.Identity()
        assert 0 <= dropout <= 1
        assert mode in self.mode
        super().__init__()
        self.resnet = resnet
        self.predictor = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(resnet.out_channels, num_classes)
        )
        
        # Regarding single label (multi-class) vs multi label classification: https://scikit-learn.org/stable/modules/multiclass.html
        # CrossEntropyLoss: softmax() in-built in loss function
        # BCEWithLogitsLoss: sigmoid() in-built in loss function; BCELoss is numerically unstable (https://github.com/pytorch/pytorch/issues/751)
        self.mode = mode
        self.loss_function = nn.MSELoss() if self.mode == "regression" else nn.CrossEntropyLoss(label_smoothing = 0.1) if self.mode == "single_label_classification" else nn.BCEWithLogitsLoss()
        self.num_classes = num_classes
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if self.optimizer is not None and optimize_predictor is True:
            self.optimizer.param_groups[0]["params"] += [*self.predictor.parameters()]

    def save(self, file_path: Union[str, os.PathLike], save_predictor_only: bool = False) -> None:
        save_model(self.predictor if save_predictor_only is True else self, file_path)

    def load(self, file_path: Union[str, os.PathLike], load_predictor_only: bool = False) -> None:
        load_model(self.predictor if load_predictor_only is True else self, file_path)

    def step(self, dataloader: DataLoader, training: bool = True) -> Tuple[Union[list, float]]:
        self.train(training) # equivalent to model.eval() if training = False
        y_true, y_pred, running_loss = [], [], 0
        for batch, (inputs, labels) in enumerate(tqdm(dataloader, leave = False, desc = "Training" if training is True else "Evaluating")):
            if training is True:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits, loss = checkpoint(self, inputs, labels, use_reentrant = False) # gradient checkpointing
            else:
                with torch.no_grad():
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    logits, loss = self(inputs, labels)
            running_loss += loss.item() * inputs.size(0)
            if self.mode == "multi_label_classification":
                predictions = (logits > 0).int() # equivalent to (torch.sigmoid(logits) > 0.5).int()
            else:
                _, predictions = torch.max(logits, 1)
            # Slight computational overhead created from returning y_true and y_pred, but it helps with visualising results immensely
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            if training is True and self.optimizer is not None:
                loss.backward() # compute gradients
                self.optimizer.step() # optimizer learns the gradients then takes a step forward
                self.optimizer.zero_grad() # set gradients to zero
        if training is True and self.lr_scheduler is not None:
            # lr_scheduler.step() is placed at the end of the episode to limit large fluctuations in lr
            if type(self.lr_scheduler) == ReduceLROnPlateau:
                self.lr_scheduler.step(running_loss)
            else:
                self.lr_scheduler.step()
        # Ground truths, predictions and average loss are returned
        return y_true, y_pred, running_loss / len(y_true)

    def forward(self, inputs: Tensor, labels: Tensor) -> Tuple[Tensor]:
        # All inputs should be tensors that are originally images
        # Regression in this case refers to predicting a continuous value based on the image given
        outputs = self.resnet(inputs).squeeze()
        logits = self.predictor(outputs)
        if self.mode == "regression":
            loss = self.loss_function(logits.squeeze(), labels.squeeze())
        elif self.mode == "single_label_classification":
            loss = self.loss_function(logits.view(-1, self.num_classes), labels.view(-1))
        elif self.mode == "multi_label_classification":
            loss = self.loss_function(logits, labels.float())
        return (logits, loss)

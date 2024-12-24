'''
This module contains the trainer script.
'''
import time
import os
import sys
import csv
import torch
import wandb
import psutil
from torch.utils.data import DataLoader
sys.path.append('../')
import dataLoader
import utils.utility


class Trainer:
    """
    Trainer class for training and evaluating a deep learning model.

    Args:
        config (dict): Configuration dictionary containing paths and model parameters.
        model (torch.nn.ModuleList): The model to be trained.
        loss_fn (torch.nn): The loss function used for training.
        log (bool, optional): Whether to log training progress. Default is False.
        project_name (str, optional): Project name for Weights & Biases logging. Default is None.
        resume (bool, optional): Whether to resume training from a checkpoint. Default is False.
        seed (int, optional): Random seed for reproducibility. Default is None.
    """
    def __init__(self, config: dict, model: torch.nn.ModuleList,
                loss_fn: torch.nn, log: bool = False, project_name: str = None,
                resume: bool = False, seed: int = None):

        self.paths = config['paths']
        self.config = config['model_configuration']
        self.meta = config['meta_data']

        self.seed = seed
        if self.seed:
            utils.utility.set_seed(self.seed)

        self.cost = loss_fn
        self.log = log
        self.project_name = project_name
        self.resume = resume
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.running_loss = None
        self.train_data, self.test = self.load_dataset()

        if self.log:
            wandb.init(
                project=self.project_name,
                config={
                    "seed": self.seed,
                    "batch size": self.config["batch_size"],
                    "number of epochs": self.config["epochs"],
                    "learning rate": self.config["learning_rate"],
                    "Model used": self.meta["model_name"] 
                }
            )

    def _run_batch(self, image, gt):
        """
        Run a single batch through the model.

        Args:
            image (torch.Tensor): Input images.
            gt (torch.Tensor): Ground truth labels.
        """
        self.optimizer.zero_grad()
        image = image.to(self.config['device'])
        gt = gt.to(self.config['device'])
        prediction = self.model(image)
        loss = self.cost(prediction, gt)

        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.detach().item()
        #print(f'\ttotal loss - {loss}\n\trunning_loss - {self.running_loss}')
        gradient_flow = utils.utility.grad_flow_dict(self.model.named_parameters())
        if self.log:
            d = {"loss": loss}
            d.update(gradient_flow)
            wandb.log(d)

    def _run_epoch(self, epoch):
        """
        Run a single epoch of training.

        Args:
            epoch (int): The current epoch number.
        """

        self.running_loss = 0
        step = 0
        epoch_start = time.time()


        for image, label in self.train_data:
            print(f"....................step - {step} of"
                  f"{len(self.train_data)}....................\n")
            batch_start = time.time()
            self._run_batch(image, label)
  
            
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            #print(f'CPU used - {cpu} ... RAM used - {ram}')
            batch_end = time.time()
            step += 1
            #print(f'\taverage_loss - {self.running_loss / step}'
            #      f'\n\tbatch time - {batch_end - batch_start}')

        epoch_end = time.time()
        if self.log:
            wandb.log({
                "loss per epoch": self.running_loss / len(self.train_data),
                "epoch time": epoch_end - epoch_start
            })
        #print(f"Epoch {epoch + 1}, Loss: {self.running_loss / len(self.train_data):.4f}"
        #      f"\ntime - {epoch_end - epoch_start}")
        
        if (epoch+1)%20==0:
            self._save_checkpoint(epoch)

        if (epoch+1)%self.config["test_every"] == 0:
            self._generate_test_output(epoch+1)

    def load_dataset(self, transform=None, shuffle=True):
        """
        Load the training and testing datasets.

        Args:
            transform (callable, optional): A function/transform to apply to the images.
            shuffle (bool, optional): Whether to shuffle the training data. Default is True.

        Returns:
            tuple: A tuple containing the training and testing data loaders.
        """
        train_data = dataLoader.CustomDataset(self.paths['dataset']['train_images'],
                                            self.paths['dataset']['train_target'],
                                            transform=transform)
        train_loader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=shuffle)
        test_data = dataLoader.CustomDataset(self.paths['dataset']['test_images'],
                                            transform=transform)
        test_loader = DataLoader(test_data, batch_size=self.config['test_batch_size'], shuffle=False)
        return train_loader, test_loader

    def _predict_test(self):
        """
        Evaluate the model on the test dataset.

        This function runs the model on the test dataset and collects the predictions.
        """
        start = time.time()
        if not self.config['device'] == 'cuda':
            self.model.eval()  # Set the model to evaluation mode
        predictions = []
        targets = []

        with torch.no_grad():
            for image, label in self.test:
                image = image.to(self.config['device'])
                output = self.model(image)
                predictions.append(output.cpu())
                targets.append(label)

        # Convert lists to tensors
        predictions = torch.cat(predictions, dim=0).detach()
        targets = [item for tup in targets for item in tup]
        stop = time.time()
        print(f'time taken to run test - {stop-start}')
        return predictions, targets
    def _write_output_file(self,predictions, file_names,output_path):
        """
        Write predictions and filenames to a CSV file.

        Args:
            predictions (torch.Tensor): Tensor containing predictions.
            fileNames (torch.Tensor): Tensor containing file names.
            output_path (str): Path to the output CSV file.
        """
        # Convert tensors to lists
        predictions_list = predictions.tolist()

        # Ensure fileNames_list contains strings
        file_names_list = [str(fileName) for fileName in file_names]

        # Write to CSV
        with open(output_path, mode='w',encoding="UTF8",newline='') as file:
            writer = csv.writer(file)
            for file_name, prediction in zip(file_names_list, predictions_list):
                writer.writerow([file_name, prediction])

    def _generate_test_output(self,epoch_num)->None:
        predictions,targets = self._predict_test()
        root = self.paths["test_prediction"]
        if not os.path.exists(root):
            os.mkdir(root)
        root = os.path.join(root,self.meta["model_name"])
        if not os.path.exists(root):
            os.mkdir(root)
        root = os.path.join(root,str(self.meta["experiment_number"]))
        if not os.path.exists(root):
            os.mkdir(root)
        root = os.path.join(root,str(epoch_num))
        if not os.path.exists(root):
            os.mkdir(root)
        root = os.path.join(root,'test_answer.csv')
        self._write_output_file(predictions,targets,root)

    def _save_checkpoint(self, epoch:int):
        """
        Save the model and optimizer state at a checkpoint.

        Args:
            n (int): The number of checkpoints to keep.
            epoch (int): The current epoch number.
        """
        root = self.paths['checkpoint']
        if not os.path.exists(root):
            os.mkdir(root)
        root = os.path.join(root, self.meta["model_name"])
        if not os.path.exists(root):
            os.mkdir(root)
        root = os.path.join(root, str(self.meta["experiment_number"]))
        if not os.path.exists(root):
            os.mkdir(root)
        torch.save(self.model.state_dict(), os.path.join(root, f'gen{epoch+1}.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(root, f'opt{epoch+1}.pth'))

    def train(self):
        """
        Train the model for the configured number of epochs.
        """
        for epoch in range(self.config['epochs']):
            self._run_epoch(epoch)

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

# custom packages
from .nn_modules import SimpleFCNN, SimpleCNN, LSTMNetwork

class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module=nn.CrossEntropyLoss,
        lr: float=1e-3
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        
        self.model = model
        self.loss_fn = loss_fn()
        self.lr = lr
    
    def process_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        x, y = batch
        if isinstance(self.model, LSTMNetwork):
            T = x.shape[-1]
            # sample a random time to get prediction
            t = np.random.randint(0, T)
            out = self.model.forward(x, predict_time=t)
        else:
            out = self.model.forward(x)
        loss = self.loss_fn(out, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=False, on_step=True, on_epoch=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        if isinstance(self.model, LSTMNetwork):
            # apparently works better for LSTM
            optimizer = torch.optim.LBFGS(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def return_model(self) -> nn.Module:
        return self.model
    

if __name__=='__main__':
    from hyper_params import lstm_params
    from nn_modules import LSTMNetwork
    from sound_data_module import SoundDataModule
    
    batch_size = 16
    spacial = (32, 96)
    
    model = LSTMNetwork(spacial, **lstm_params.get('model_kwargs'))
    training_model = TrainingModule(model=model, **lstm_params.get('training_module_kwargs'))
    
    data_module = SoundDataModule('./data')
    data_module.setup('fit')
    data_loader = data_module.train_dataloader()._get_iterator()
    x, y = next(data_loader)
    
    # x = torch.randn((batch_size, 1, *spacial))
    # y = torch.randint(low=0, high=5, size=(batch_size, ))
    
    loss = training_model.training_step((x, y), None)
    print(f'{loss=}')
    print('Success')
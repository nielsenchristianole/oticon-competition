import os
import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from oticon_utils.hyper_params import fcnn_params, cnn_params, lstm_params
from oticon_utils.nn_modules import SimpleFCNN, SimpleCNN, LSTMNetwork
from oticon_utils.sound_data_module import SoundDataModule
from oticon_utils.training_module import TrainingModule

params_dict = dict(
    fcnn=fcnn_params,
    cnn=cnn_params,
    lstm=lstm_params
)
models_dict = dict(
    fcnn=SimpleFCNN,
    cnn=SimpleCNN,
    lstm=LSTMNetwork
)

def main(model_type: str, epochs: int, seed: int=None, device: str='cuda'):
    models_dir = os.path.join('./models/', f'{model_type}-{seed}')
    
    params = params_dict[model_type]
    
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    
    one_hot = params.get('training_module_kwargs').get('loss_fn') is torch.nn.MSELoss
    # get dataloaders
    sound_context_lenght = params.get('sound_context_lenght')
    data_module = SoundDataModule('./data/', sound_context_lenght=sound_context_lenght, one_hot=one_hot, balance=True, class_subset=[2, 4])
    assert device=='cpu' or torch.cuda.is_available(), "Cuda is not available, please select cpu as device"
    
    # get model
    input_dims = (32, sound_context_lenght if sound_context_lenght > 0 else 96)
    model_kwargs = params.get('model_kwargs')
    model = models_dict[model_type](input_dims=input_dims, **model_kwargs)
    
    # get training model
    training_module_kwargs = params.get('training_module_kwargs')
    training_model = TrainingModule(model, **training_module_kwargs).to(device)
    
    loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=5, filename='loss-{epoch}-{val_loss:.3}')
    
    # init trainer
    trainer = pl.Trainer(
        default_root_dir=models_dir,
        callbacks=[loss_callback],
        max_epochs=epochs,
        logger=CSVLogger(models_dir, flush_logs_every_n_steps=100),
        log_every_n_steps=10,
        accelerator=device
    )

    # train loop
    trainer.fit(
        training_model,
        data_module
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--epochs')
    parser.add_argument('--seed')
    parser.add_argument('--device')
    
    args = parser.parse_args()
    
    model_type = 'cnn' if args.model is None else args.model
    epochs = 20 if args.epochs is None else args.epochs
    seed = args.seed
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert torch.cuda.is_available()
    
    main(model_type, epochs, seed, device)
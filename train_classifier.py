import os
import argparse
import pickle
import glob
import numpy as np

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

def main(model_type: str, epochs: int, seed: int=None, device: str='cuda', work_dir='./'):
    models_dir = os.path.join(work_dir, 'models/', f'{model_type}-{seed}')
    data_dir = os.path.join(work_dir, 'data/')
    
    params = params_dict[model_type]
    
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    
    one_hot = params.get('training_module_kwargs').get('loss_fn') is torch.nn.MSELoss
    # get dataloaders
    sound_context_lenght = params.get('sound_context_lenght')
    data_module = SoundDataModule(data_dir, sound_context_lenght=sound_context_lenght, one_hot=one_hot, balance=3.)
    assert device=='cpu' or torch.cuda.is_available(), "Cuda is not available, please select cpu as device"
    loss_weights = data_module.get_loss_weight()
    
    # get model
    input_dims = (32, sound_context_lenght if sound_context_lenght > 0 else 96)
    model_kwargs = params.get('model_kwargs')
    model = models_dict[model_type](input_dims=input_dims, **model_kwargs)
    
    # get training model
    training_module_kwargs = params.get('training_module_kwargs')
    training_model = TrainingModule(model, loss_weights=loss_weights, **training_module_kwargs).to(device)
    
    loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=5, filename='loss-{epoch}-{val_loss:.3}-acc-{val_acc:.3}')
    
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
    
    # get precictions and do eval metrics
    trainer.test(
        training_model,
        data_module,
        ckpt_path='best'
    )
    
    version_path = os.path.join(models_dir, 'lightning_logs', 'version_0')
    v = 0
    while True:
        v += 1
        path = os.path.join(models_dir, 'lightning_logs', f'version_{v}')
        if os.path.exists(path):
            version_path = path
        else:
            break
    
    test_predictions, test_labels = training_model.return_test_results()
    np.save(os.path.join(version_path, 'val_predictions.npy'), test_predictions)
    np.save(os.path.join(version_path, 'val_labels.npy'), test_labels)
    
    predict_out: list[torch.Tensor]
    predict_out = trainer.predict(
        training_model,
        data_module,
        ckpt_path='best'
    )
    predict_out = [t.numpy() for t in predict_out]
    predict_out: list[np.ndarray]
    predict_out = np.concatenate(predict_out)
    predict_out: np.ndarray
    np.save(os.path.join(version_path, 'test_predictions.npy'), predict_out)
    predict_out = np.argmax(predict_out, axis=1)
    np.savetxt(os.path.join(version_path, 'test_predictions.txt'), predict_out, fmt='%s')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str)
    parser.add_argument('--epochs', type = int)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--device', type = str)
    
    args = parser.parse_args()
    
    model_type = 'cnn' if args.model is None else args.model
    epochs = 25 if args.epochs is None else args.epochs
    seed = args.seed
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert torch.cuda.is_available()
    
    main(model_type, epochs, seed, 'cpu', work_dir=r'C:\Users\niels\local_data\oticon')
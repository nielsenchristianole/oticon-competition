
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
import numpy as np

import torch.nn.utils.prune as prune

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


def compute_prune_amount(iteration):
    return (np.exp(0.4/(iteration+1)) - 1)/1.3
            
def main(model_type: str, epochs: int, prune_epochs: int, interations: int, seed: int=None, device: str='cuda'):
    models_dir = os.path.join('./models/', f'{model_type}-{seed}')
    
    params = params_dict[model_type]
    
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    
    one_hot = params.get('training_module_kwargs').get('loss_fn') is torch.nn.MSELoss
    # get dataloaders
    sound_context_lenght = params.get('sound_context_lenght')
    data_module = SoundDataModule('./data/', sound_context_lenght=sound_context_lenght, one_hot=one_hot, balance=3.)
    assert device=='cpu' or torch.cuda.is_available(), "Cuda is not available, please select cpu as device"
    
    # get model
    input_dims = (32, sound_context_lenght if sound_context_lenght > 0 else 96)
    model_kwargs = params.get('model_kwargs')
    model = models_dict[model_type](input_dims=input_dims, **model_kwargs)
    
    # get training model
    training_module_kwargs = params.get('training_module_kwargs')
    training_model = TrainingModule(model, **training_module_kwargs).to(device)
    
    loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=5, filename='loss-{epoch}-{val_loss:.3}-{val_acc:.3}')

    
    # # Writting as lambda function to ensure compatability with pickle
    # compute_prune_amount = lambda epoch : torch.exp(0.4/idx) - 1 if( idx := epoch - epochs <= 0 and (idx - 1)%3 == 0) else 0

    print(">>>", type(epochs), type(prune_epochs))
    # init trainer
    init_trainer = pl.Trainer(
        default_root_dir=models_dir,
        callbacks=[loss_callback],
        max_epochs=epochs,
        logger=CSVLogger(models_dir, flush_logs_every_n_steps=100),
        log_every_n_steps=10,
        accelerator=device
    )
    
    
    
    ### Initial training
    
    # train loop
    init_trainer.fit(
        training_model,
        data_module
    )
    
    # get precictions and do eval metrics
    init_trainer.test(
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
    np.save(os.path.join(version_path, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(version_path, 'test_labels.npy'), test_labels)
    
    
    loss_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=5, filename='(pruned)loss-{epoch}-{val_loss:.3}-{val_acc:.3}')


    ### Do 'iterations' number of prunning.
    # Detach the pruning from pytorch lightning due to issues
    for iter in range(interations):
        
        pruned_trainer = pl.Trainer(
            default_root_dir=models_dir,
            callbacks=[loss_callback],
            max_epochs=prune_epochs,
            logger=CSVLogger(models_dir, flush_logs_every_n_steps=100),
            log_every_n_steps=10,
            accelerator=device
        )
        
        parameters_to_prune = []
        for module_name, module in training_model.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                for parameter in module.parameters():
                    parameter.detach()
                    parameter.requires_grad_(requires_grad = False)
                parameters_to_prune.append((module, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=compute_prune_amount(iter),
        )
        
        n_pruned_param = 0
        for name, parameters in model.named_parameters():
            if(name.endswith('_orig')):
                n_pruned_param += parameters.numel()
        
        print(f">>> Iteration: {iter}, Total pruned: {n_pruned_param}")
        # train loop
        pruned_trainer.fit(
            training_model,
            data_module
        )
        
    # get precictions and do eval metrics
    pruned_trainer.test(
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
    np.save(os.path.join(version_path, 'test_predictions.npy'), test_predictions)
    np.save(os.path.join(version_path, 'test_labels.npy'), test_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--prune_epochs', type=int)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--iterations',type=int,default=0)
    
    args = parser.parse_args()
    
    model_type = 'cnn' if args.model is None else args.model
    epochs = 0 if args.epochs is None else args.epochs
    prune_epochs = 6 if args.prune_epochs is None else args.prune_epochs
    seed = args.seed
    device = args.device
    iterations = args.iterations
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert torch.cuda.is_available()
    
    main(model_type, epochs, prune_epochs, iterations, seed, 'cpu')
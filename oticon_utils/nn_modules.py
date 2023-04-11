import tabulate

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class SimpleCNN(nn.Module):
    """
    Convolutional neural network
    """
    def __init__(
        self,
        channels: list,
        fc_dims: list,
        dropout: float=0.,
        channels_layer_repeats: int=3,
        input_dims: list[int, int]=(32, 96),
    ):
        super().__init__()
        
        self.channels = channels
        self.fc_dims = fc_dims
        self.dropout = dropout
        self.channels_layer_repeats = channels_layer_repeats
        self.input_dims=input_dims
        
        convolution_layers = list()
        for idx_channel, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            layer = [nn.InstanceNorm2d(in_channels)]
            for _ in range(channels_layer_repeats):
                layer.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same'),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
                in_channels = out_channels
            if idx_channel != len(channels)-1:
                # don't pool the lasy layer
                layer.append(nn.MaxPool2d((2,2)))
            convolution_layers.append(nn.Sequential(*layer))
        self.convolution_layers = nn.ModuleList(convolution_layers)
        
        # add input dims from convolutions
        fc_dims = [np.prod(input_dims) // (4 ** (len(channels) - 1)) * channels[-1]] + list(fc_dims)
        dense_layers = list()
        for in_dim, out_dim in zip(fc_dims[:-2], fc_dims[1:-1]):
            dense_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        dense_layers.append(nn.Linear(fc_dims[-2], fc_dims[-1]))
        
        self.dense_layers = nn.ModuleList(dense_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.convolution_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)
        for layer in self.dense_layers:
            x = layer(x)
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return F.softmax(x, dim=-1)
    
    
class SimpleRNN(nn.Module):
    """
    Recurrent neural network
    """
    def __init__(self):
        super().__init__()


class SimpleFCNN(nn.Module):
    """
    Fully connected neural network
    dims: (in, [hidden, ...], out)
    """
    def __init__(self, dims: list, dropout: float=0., self_flatten: bool=True):
        super().__init__()
        self.dims = dims
        self.dropout = dropout
        self.self_flatten = self_flatten
        
        layers = list()
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.self_flatten:
            x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return F.softmax(x, dim=-1)

if __name__ == '__main__':
    img_size = (32, 96)
    x = torch.randn(16, 1, *img_size)
    
    # model = SimpleFCNN((np.prod(img_size), 16, 5), 0.1)
    model = SimpleCNN((1, 2, 4), (128, 128, 5))
    print(model.predict(x).shape)

    
    tabular_data = list()
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        tabular_data.append((name, params))
        total_params += params
    tabular_data.append(('Total', total_params))
    
    params_table = tabulate.tabulate(tabular_data, ('Parameter', 'Count'))
    print(params_table)
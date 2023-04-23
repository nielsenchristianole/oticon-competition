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
        in_channels: int=1,
        channel_mult: int=32,
        dropout: float=0.,
        channels_layer_repeats: int=3,
        input_dims: list[int, int]=(32, 96),
    ):
        super().__init__()
        
        self.channels = channels
        self.fc_dims = fc_dims
        self.dropout = dropout
        self.channels_layer_repeats = channels_layer_repeats
        self.input_dims = input_dims
        self.in_channels = in_channels
        self.channel_mult = channel_mult
        
        convolution_layers = [
            nn.Conv2d(in_channels, channel_mult * channels[0], kernel_size=(3, 3), padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        for idx_channel, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            layer = [nn.InstanceNorm2d(in_channels)]
            for _ in range(channels_layer_repeats):
                layer.append(
                    nn.Sequential(
                        nn.Conv2d(channel_mult * in_channels, channel_mult * out_channels, kernel_size=(3, 3), padding='same'),
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
        fc_dims = [np.prod(input_dims) * channels[-1] * channel_mult // (4 ** (len(channels) - 1))] + list(fc_dims)
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
        # if len(x.shape) == 4:
        #     pass
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        for layer in self.convolution_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)
        for layer in self.dense_layers:
            x = layer(x)
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return F.softmax(x, dim=-1)


class LSTMNetwork(nn.Module):
    """
    Recurrent neural network build with long short term memory
    
    Notation follows https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
    """
    def __init__(
        self,
        input_dims: tuple[int, int],
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()
        
        freq_dim, sound_context_length = input_dims
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.f_t = nn.Sequential(
            nn.Linear(freq_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.i_t = nn.Sequential(
            nn.Linear(freq_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.c_plus = nn.Sequential(
            nn.Linear(freq_dim + hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.o_t = nn.Sequential(
            nn.Linear(freq_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_dim, output_dim)
        
        self.h_layer_norm = nn.LayerNorm(hidden_dim)
        self.c_layer_norm = nn.LayerNorm(hidden_dim)


    def forward(
        self,
        x: torch.Tensor,
        memory: tuple[torch.Tensor, torch.Tensor]=None,
        predict_time: int=None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | torch.Tensor:
        """
        Memory: the candidate and hidden state of the RNN
        predict_time: at what timestep to return the prediction
        """
        if len(x.shape) == 4:
            batch_dim, channel_dim, freq_dim, sequence_dim = x.shape
            assert channel_dim == 1, f"Input shape {x.shape=} does not follow assumed dim"
            x = x.squeeze(1)
        elif len(x.shape) == 3:
            batch_dim, freq_dim, sequence_dim = x.shape
        else:
            raise ValueError(f'invalid shape of {x.shape=}')
        
        if memory is None:
            h_t = torch.zeros((batch_dim, self.hidden_dim)).to(x.device)
            c_t = torch.zeros((batch_dim, self.hidden_dim)).to(x.device)
        else:
            h_t, c_t = memory
        
        outputs = list()
        for t in range(sequence_dim):
            x_t = x[..., t]
            
            combined = torch.cat((x_t, h_t), dim=-1)
            
            c_t = c_t * self.f_t(combined) + self.i_t(combined) * self.c_plus(combined)
            h_t = self.o_t(combined) * self.tanh(c_t)
            
            # normelize
            c_t, h_t = self.c_layer_norm(c_t), self.h_layer_norm(h_t)
            
            out = self.out(h_t)
            if t == predict_time:
                return out
            outputs.append(out.unsqueeze(-1))
        assert predict_time is None, f'{predict_time=}, is not understood, select int below {len(sequence_dim)}'
        
        outputs = torch.cat(outputs, dim=-1)
        
        return outputs, (h_t, c_t)
    
    def predict(
        self,
        x: torch.Tensor,
        memory: tuple[torch.Tensor, torch.Tensor]=None,
        predict_time: int=None
    ):
        assert self.training, "Model not in training mode"
        if predict_time is None:
            predict_time = x.shape[-1] - 1
        x = self.forward(x, memory=memory, predict_time=predict_time)
        return F.softmax(x, dim=-1)
        


class SimpleFCNN(nn.Module):
    """
    Fully connected neural network
    input_dims: dim of img before flattening
    dims: ([hidden, ...], out)
    """
    def __init__(
        self,
        input_dims: tuple[int, int],
        dims: list,
        dropout: float=0.,
        self_flatten: bool=True
    ):
        super().__init__()
        self.input_dims = input_dims
        self.dims = dims = [np.prod(input_dims)] + list(dims)
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
    from hyper_params import lstm_params
    from nn_modules import LSTMNetwork
    
    img_size = (32, 96)
    n_classes = 5
    batch_size = 16
    
    x = torch.randn(batch_size, 1, *img_size)
    
    
    # model = SimpleFCNN((np.prod(img_size), 16, n_classes), 0.1)
    model = SimpleCNN((1, 2, 4), (128, 128, n_classes))
    # model = LSTMNetwork(img_size, **lstm_params.get('model_kwargs'))
    # model = LSTMNetwork(img_size[0], 16, n_classes)
    out = model.forward(x, predict_time=10)
    print(out.shape)
    # print(out[0].shape, out[1][0].shape, out[1][1].shape)

    
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
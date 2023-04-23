from torch.nn import CrossEntropyLoss, MSELoss

fcnn_params = dict(
    sound_context_lenght=-1,
    model_kwargs=dict(
            dims=(512, 256, 128, 64, 5),
            dropout=0.
    ),
    training_module_kwargs=dict(
        loss_fn=CrossEntropyLoss,
        lr=1e-3,
    )
)

cnn_params = dict(
    sound_context_lenght=-1,
    model_kwargs=dict(
        channels=(1, 2, 4, 8), # len = number of contractive layers; int = channel multiplier for the layer
        fc_dims=(32,64,5,), # output dims for each dense layer, 5 is number of classes
        in_channels=1, # number of input channels of image (spectrogram is single channel)
        channel_mult=12, # base number of channels to be multiplied by channels
        dropout=0.3,
        channels_layer_repeats=3, # number of conv layers for each layer
    ),
    training_module_kwargs=dict(
        loss_fn=CrossEntropyLoss,
        lr=1e-3,
        weight_decay=1e-5
    )
)

lstm_params = dict(
    sound_context_lenght=-1,
    model_kwargs=dict(
        hidden_dim=256,
        output_dim=5
    ),
    training_module_kwargs=dict(
        loss_fn=MSELoss,
        lr=0.08 # uses LBFGS optimizer
    )
)
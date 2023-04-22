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
        channels=(1, 2, 4, 8),
        fc_dims=(64,64,5),
        in_channels=1,
        channel_mult=32,
        dropout=0.6,
        channels_layer_repeats=3,
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
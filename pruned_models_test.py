
from pytorch_lightning import LightningModule
from oticon_utils.training_module import TrainingModule

model = TrainingModule.load_from_checkpoint("models/cnn-None/lightning_logs/version_30/checkpoints/loss-epoch=0-val_loss=1.55-val_acc=0.249.ckpt")

for name, parame in model.model.parameters():
    print(name)
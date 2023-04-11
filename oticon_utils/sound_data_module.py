import os

import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, TensorDataset, DataLoader


class SoundDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        *,
        batch_size: int=64,
        sound_snippet_lenght: int=-1,
        train_val_split: tuple[float, float] = (0.8, 0.2),
        num_workers: int=None
    ):
        """
        Datamodule for oticon challenge.
        
        sound_snippet_lenght: lenght of sound snippets. Set to -1 for returning the whole audio
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sound_snippet_lenght = sound_snippet_lenght
        self.train_val_split = train_val_split
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        
        assert sound_snippet_lenght <= 96, f'{sound_snippet_lenght=} has to be lower than 96'
        assert np.sum(train_val_split) == 1., f'{train_val_split=} has to sum to 1'
    
    def setup(self, stage: str):
        if stage == 'fit':
            X = torch.Tensor(np.load(os.path.join(self.data_dir, 'training.npy')))
            y = torch.Tensor(np.load(os.path.join(self.data_dir, 'training_labels.npy')))
            whole_dataset = TensorDataset(X, y)
            self.train_set, self.val_set = random_split(whole_dataset, self.train_val_split)
            
        elif stage == 'predict':
            X = torch.Tensor(np.load(os.path.join(self.data_dir, 'test.npy')))
            self.predict_set = TensorDataset(X)
        else:
            raise NotImplementedError(f'{stage=} not implemented')
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate_X_y)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate_X_y)

    def test_dataloader(self):
        raise NotImplementedError('test_dataloader has not been implemented')

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate_X)
    
    def _collate_X_y(self, batch):
        X, y = list(zip(*batch))

        if self.sound_snippet_lenght == -1:
            return torch.stack(X), torch.stack(y)
        
        return_X = list()
        return_y = list()
        for X, y in zip(X, y):
            start = np.random.randint(0, X.shape[-1] - self.sound_snippet_lenght)
            end = start + self.sound_snippet_lenght
            return_X.append(X[..., start:end])
            return_y.append(y)

        return torch.stack(return_X), torch.stack(return_y)
    
    def _collate_X(self, batch):
        X = list(zip(*batch))[0]

        if self.sound_snippet_lenght == -1:
            return torch.stack(X)
        
        return_X = list()
        for X in X:
            start = np.random.randint(0, X.shape[-1] - self.sound_snippet_lenght)
            end = start + self.sound_snippet_lenght
            return_X.append(X[..., start:end])

        return torch.stack(return_X)

if __name__ == "__main__":
    # testing to see if it works
    data_dir = './data/'
    data_module = SoundDataModule(data_dir, sound_snippet_lenght=-1)
    data_module.setup('fit')
    data_module.setup('predict')

    train_loader = data_module.train_dataloader()._get_iterator()
    predict_loader = data_module.predict_dataloader()._get_iterator()
    
    for _ in range(3):
        X, y = next(train_loader)
        print(X.shape)
        # print(X)
        print(y.shape)
        # print(y)
    
    for _ in range(3):
        X = next(predict_loader)
        print(X.shape)
    
    print('Success', data_module)

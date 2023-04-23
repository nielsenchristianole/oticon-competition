import os

import numpy as np

import torch
from torch.utils.data import random_split, TensorDataset, DataLoader
import pytorch_lightning as pl


class SoundDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        *,
        batch_size: int=64,
        sound_context_lenght: int=-1,
        one_hot: bool=False,
        num_classes: int=5,
        train_val_split: tuple[float, float] = (0.8, 0.2),
        num_workers: int=None,
        balance: float=0.,
        class_subset: list=None,
        data_split_seed: int=42,
    ):
        """
        Datamodule for oticon challenge.
        
        sound_snippet_lenght: lenght of sound snippets. Set to -1 for returning the whole audio
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sound_snippet_lenght = sound_context_lenght
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.train_val_split = train_val_split
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.balance = balance
        self.data_split_seed = data_split_seed
        self.numpy_X = None
        self.numpy_y = None
        
        if class_subset is None:
            class_subset = list(range(num_classes))
        self.class_subset = class_subset
        
        assert sound_context_lenght <= 96, f'{sound_context_lenght=} has to be lower than 96'
        assert np.sum(train_val_split) == 1., f'{train_val_split=} has to sum to 1'
    
    def setup(self, stage: str):
        if stage == 'fit':
            X, y = np.load(os.path.join(self.data_dir, 'training.npy')), np.load(os.path.join(self.data_dir, 'training_labels.npy'))
            Xs, ys = list(), list()
            min_num_classes = float('inf')
            for label in self.class_subset:
                count = np.sum(y == label)
                min_num_classes = count if count < min_num_classes else min_num_classes
            min_num_classes = int(min_num_classes * self.balance)
            for label in self.class_subset:
                mask = (y == label)
                tmp = X[mask], y[mask]
                if self.balance > 0:
                    count = min(np.sum(y == label), min_num_classes)
                    choices = np.random.choice(len(tmp[1]), size=count, replace=False)
                    tmp = tmp[0][choices], tmp[1][choices]
                Xs.append(tmp[0])
                ys.append(tmp[1])
            X, y = np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
            self.numpy_X, self.numpy_y = X, y
            X, y = torch.Tensor(X), torch.Tensor(y)
            whole_dataset = TensorDataset(X, y)
            self.train_set, self.val_set = random_split(whole_dataset, self.train_val_split, generator=torch.Generator().manual_seed(self.data_split_seed))
        elif stage == 'test':
            pass
        elif stage == 'predict':
            X = torch.Tensor(np.load(os.path.join(self.data_dir, 'test.npy')))
            self.predict_set = TensorDataset(X)
        else:
            raise NotImplementedError(f'{stage=} not implemented')
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self._collate_X_y)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate_X_y)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=self._collate_X)
    
    def get_loss_weight(self) -> torch.Tensor:
        if self.numpy_y is None:
            self.setup('fit')
        class_distribution = list()
        for label in self.class_subset:
            class_distribution.append(np.sum(self.numpy_y == label) / len(self.numpy_y))
        class_distribution = np.array(class_distribution)
        class_distribution = 1 / class_distribution
        class_distribution = class_distribution * len(self.class_subset) / sum(class_distribution)
        return torch.from_numpy(class_distribution).to(torch.float32)
    
    def _collate_X_y(self, batch):
        X, y = list(zip(*batch))
        y = torch.stack(y).to(int)
        
        if self.one_hot:
            y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).to(torch.float32)
        
        if self.sound_snippet_lenght == -1:
            return torch.stack(X), y
        
        return_X = list()
        return_y = list()
        for X, y in zip(X, y):
            start = np.random.randint(0, X.shape[-1] - self.sound_snippet_lenght)
            end = start + self.sound_snippet_lenght
            return_X.append(X[..., start:end])
            return_y.append(y)

        return torch.stack(return_X), torch.stack(return_y).squeeze().to(int)
    
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
    data_module = SoundDataModule(data_dir, sound_context_lenght=-1, one_hot=False)
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

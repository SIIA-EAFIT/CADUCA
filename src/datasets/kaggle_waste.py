from typing import Optional, Union, Callable
import shutil
import pathlib

from kaggle import KaggleApi
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl

class WasteDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root_dir: Union[str, pathlib.Path],
            batch_size: int,
            transform: Optional[Callable] = None,
            train_size: Union[float, int] = 0.7,
            num_workers: int = 0,
    ):
        super().__init__()
        self._data_root_dir = pathlib.Path(data_root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train_size = train_size

        self.transforms = transform

        self.idx_to_class = ['Organic', 'Recyclable']
    
    def prepare_data(self):
        """
        Things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode, s.a. download.
        """
        if not self._data_root_dir.exists():
            kaggle = KaggleApi()
            kaggle.authenticate()
    
            kaggle.dataset_download_files(
                'techsash/waste-classification-data',
                path=self._data_root_dir,
                quiet=False,
                unzip=True
            )

            # post processing
            shutil.rmtree(self._data_root_dir / 'dataset')

            tmp_dir = (self._data_root_dir / 'DATASET')

            for split in tmp_dir.iterdir():
                split.rename(self._data_root_dir / split.name)

                tmp_dir.rmdir()

    def setup(self, _):
        """ 
        Things to do on every accelerator in distributed mode i.e. make assignments
        """
        whole_train_dataset = ImageFolder(
            self._data_root_dir / 'TRAIN',
            transform=self.transforms
        )

        dataset_size = len(whole_train_dataset)
        
        if isinstance(self._train_size, int):
            if dataset_size < train_size:
                raise ValueError('train size {train_size} is larger than {dataset_size}')
            train_size = self._train_size
        elif isinstance(self._train_size, float):
            if self._train_size < 1:
                train_size = int(self._train_size * dataset_size)
            else:
                raise ValueError('train size {train_size} is larger than 1')
            
        split_size = [train_size, dataset_size - train_size]
        self._train_dataset, self._val_dataset = random_split(
            whole_train_dataset, split_size
        )
        self._test_dataset = ImageFolder(
            self._data_root_dir / 'TEST',
            transform=self.transforms
        )
        
        
    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers
        )

    

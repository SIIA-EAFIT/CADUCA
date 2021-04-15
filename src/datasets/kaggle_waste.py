from typing import Optional
import shutil
import pathlib

from kaggle import KaggleApi
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

def prepare_data(data_root_dir: pathlib.Path):
    kaggle = KaggleApi()
    kaggle.authenticate()
    
    kaggle.dataset_download_files(
        'techsash/waste-classification-data',
        path=data_root_dir,
        quiet=False,
        unzip=True
    )

    # post processing
    shutil.rmtree(data_root_dir / 'dataset')

    tmp_dir = (data_root_dir / 'DATASET')

    for split in tmp_dir.iterdir():
        split.rename(data_root_dir / split.name)

    tmp_dir.rmdir()

def get_waste_dataset(
        data_root_dir: Optional[str] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
) -> Dataset:
    if data_root_dir is None:
        root_dir = pathlib.Path(__file__).absolute().parents[2]

        data_root_dir = root_dir / 'data' / 'kaggle-waste-data'
        data_root_dir.mkdir(exist_ok=True, parents=True)
    else:
        data_root_dir = pathlib.Path(data_root_dir)

    if not data_root_dir.exists():
        prepare_data(data_root_dir)

    if train:
        root = data_root_dir / 'TRAIN'
    else:
        root = data_root_dir / 'TEST'

    return ImageFolder(
        root, transform, target_transform
    )
        

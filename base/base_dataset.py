import numpy as np
from pathlib import Path
from abc import abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base class for all datasets
    """
    def __init__(self, root_dir):
        """
        :param root_dir: Directory with all the data
        Load data as numpy array
        """
        self.root_dir = Path(root_dir)

    @abstractmethod
    def __getitem__(self, index):
        """
        Support the indexing of the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Return the size of the dataset
        """
        raise NotImplementedError


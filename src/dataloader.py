from typing import Dict
import os
from torchvision import datasets

from src.image_augmentations import CIFAR10Transform, BasicTransform
root_dir = os.path.join(os.path.dirname(__file__), './data/')


def get_data_manager(data_config: Dict):
    """
        @param data_config: pass data configuration.
        example data config
        {
            "data_set": "cifar10",
            "train_batch_size": 128,
            "test_batch_size": 1000,
            "num_worker": 6,
        },
        @return: Data Manager object
    """
    return DataManager(data_config=data_config)

class DataManager:
    """
    @param data_config = config
    """

    def __init__(self,
                 data_config: Dict):
        self.data_config = data_config

        self.data_set = self.data_config.get('data_set')
        self.train_batch_size = self.data_config.get('train_batch_size', 256)
        self.test_batch_size = self.data_config.get('test_batch_size', 1000)
        self.num_worker = self.data_config.get('num_worker', 1)

        # initialize attributes specific to dataset
        self.num_channels, self.height, self.width = None, None, None
        self.mean, self.std = None, None
        self.tr_dataset, self.te_dataset = None, None
        self.model_ip_shape = None
        self.sv_transform, self.mv_transform, self.basic_transform = None, None, None

    def get_dataset(self):
        """
        Returns:
            train and test dataset
        """
        if self.data_set == 'cifar10':
            # update dataset attributes
            self.num_channels, self.height, self.width = 3, 32, 32
            self.mean, self.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            self.model_ip_shape = 32

            # get datasets
            tr_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True)
            te_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True)

            # get image transforms - mv: multi-viewed, sv: single-viewed, basic: just normalize
            self.mv_transform = CIFAR10Transform(mean=self.mean, std=self.std, multi_viewed=True, input_shape=self.model_ip_shape)
            self.sv_transform = CIFAR10Transform(mean=self.mean, std=self.std, multi_viewed=False, input_shape=self.model_ip_shape)

        else:
            raise NotImplementedError

        self.basic_transform = BasicTransform(mean=self.mean, std=self.std, input_shape=self.model_ip_shape)

        return tr_dataset, te_dataset

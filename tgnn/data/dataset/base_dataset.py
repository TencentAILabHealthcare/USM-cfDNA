# Copyright (c) 2024, Tencent Inc. All rights reserved.

from torch.utils.data import Dataset


class BaseDataset(Dataset):

    @classmethod
    def build_train_dataset(cls, cfg):
        raise NotImplementedError

    @classmethod
    def build_val_dataset(cls, cfg):
        raise NotImplementedError

    @classmethod
    def build_test_dataset(cls, cfg):
        raise NotImplementedError

    @classmethod
    def build_dataset(cls, cfg):
        raise NotImplementedError

    def __int__(self, *args, **kwargs):
        super(BaseDataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"#samples: {len(self)}"

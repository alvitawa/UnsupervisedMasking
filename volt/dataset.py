from torch.utils.data import Dataset


class SliceableDataset(Dataset):
    def __init__(self, dataset=None, slice=None):
        super(SliceableDataset, self).__init__()
        self.__dataset = dataset
        self.__slice = slice

    def get_item(self, index):
        raise NotImplementedError('You must override get_item')

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SliceableDataset(self, index)
        elif self.__dataset is None:
            return self.get_item(index)
        else:
            start, stop, step = self.__slice.indices(len(self.__dataset))
            if index < 0:
                index += len(self)
            assert 0 <= index < len(self), 'Index out of range'
            return self.__dataset[start + index * step]

    def __len__(self):
        assert self.__dataset is not None, 'You must override __len__'
        start, stop, step = self.__slice.indices(len(self.__dataset))
        return (stop - start) // step

    def __getattr__(self, attr):
        if self.__dataset is None:
            raise AttributeError(f'{self.__class__.__name__} object has no attribute {attr}')
        return getattr(self.__dataset, attr)

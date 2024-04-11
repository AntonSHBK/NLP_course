import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features, targets):
        """
        Инициализация датасета.
        :param features: Numpy массив или список списков с признаками.
        :param targets: Numpy массив или список с целевыми значениями.
        """
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Получение одного элемента датасета по индексу.
        """
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)
    
    def get_features_targets(self, idx):
        return self.features[idx], self.targets[idx]

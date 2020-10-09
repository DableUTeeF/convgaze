import os
from torch.utils.data import Dataset
import random


class Datagen:
    def __init__(self, val_split=0.2):
        self.val_split = val_split
        self.list = []
        for x in os.listdir('output'):
            if 'txt' in x:
                self.list.append(x)
        random.shuffle(self.list)

    class EyesDataset(Dataset):
        def __init__(self, list):
            self.list = list

        def __getitem__(self, idx):
            x = self.list[idx].split('_')[0]
            for line in open(os.path.join('output', self.list[idx])).read().split('\n'):
                if line == '':
                    break


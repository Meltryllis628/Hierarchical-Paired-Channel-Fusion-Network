from torch.utils.data import Dataset

class TSUNAMIDataset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def add_data(self, item):
        self.data.append(item)
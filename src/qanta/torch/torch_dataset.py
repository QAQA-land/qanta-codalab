from torch.utils.data import Dataset
from qanta.dataset import QuizBowlDataset

class TorchQBData(Dataset):
    def __init__(self, dataset: QuizBowlDataset):
        self.qs, self.pages, _ = dataset.training_data()
        assert(len(self.qs) == len(self.pages))
    
    def __len__(self):
        return(len(self.qs))

    def __getitem__(self, idx):
        sample = {'text': self.qs[idx], 'page': self.pages[idx]}
        return sample
import torch
from torch.utils.data import DataLoader, TensorDataset

class AtkDataSet():
    def __init__(self, attack_batches, test_loader):
        self.attack_batches = attack_batches
        self.test_loader = test_loader
    
    def add_query(self, query: torch.Tensor, response: torch.Tensor):
        if not hasattr(self,'queries'):
            self.queries = query
            self.response = response
        else:

            self.queries = torch.cat([self.queries, query])
            self.response = torch.cat([self.response, response])

    def load_atk_dataset(self):
        return DataLoader(TensorDataset(self.queries, self.response), batch_size=self.attack_batches, shuffle=True)
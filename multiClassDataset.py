import torch
from torch.utils.data import TensorDataset

# Classe pour regrouper plusieurs datasets avec des étiquettes de classe
class MultiClassDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_dict):
        self.datasets = []
        self.labels = []
        self.class_names = list(datasets_dict.keys())

        # Associer les noms de classes à des indices
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Combiner tous les datasets
        for class_name, dataset in datasets_dict.items():
            if isinstance(dataset, torch.utils.data.TensorDataset):
                data = dataset.tensors[0]  # Accéder au premier tenseur du TensorDataset
            else:
                # Sinon, on suppose que c'est déjà un tenseur ou un format compatible
                data = dataset

            class_idx = self.class_to_idx[class_name]
            self.datasets.append(data)
            # Créer un tenseur d'étiquettes de la même longueur que les données
            self.labels.append(torch.full((len(data),), class_idx, dtype=torch.long))

        self.data = torch.cat(self.datasets, dim=0)
        self.label = torch.cat(self.labels, dim=0)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

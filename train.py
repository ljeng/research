import pathlib
import const
import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class Trainer:
    def __init__(self):
        self.model = nn.Sequential()
        self.model.add_module('layers',
            nn.Sequential(nn.Flatten(),
            nn.Linear(const.IN_FEATURES, const.FEATURES),
            nn.ReLU(),
            nn.Linear(const.FEATURES, len(const.CLASSES)))
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters())
        self.history = []

    def train(self, loader):
        self.model.train()
        sum_loss = count_eq = size = 0
        for images, labels in loader:
            self.optim.zero_grad()
            models = self.model(images)
            cross_entropy_loss = self.cross_entropy_loss(models, labels)
            item = cross_entropy_loss.item()
            cross_entropy_loss.backward()
            self.optim.step()
            size += labels.size(0)
            sum_loss += item * labels.size(0)
            count_eq += (models.argmax(dim=1) == labels).sum().item()
        return {'loss': sum_loss / size, 'eq': count_eq / size}

    @torch.inference_mode()
    def evaluate(self, loader):
        self.model.eval()
        sum_loss = count_eq = size = 0
        for images, labels in loader:
            models = self.model(images)
            cross_entropy_loss = self.cross_entropy_loss(models, labels)
            item = cross_entropy_loss.item()
            size += labels.size(0)
            sum_loss += item * labels.size(0)
            count_eq += (models.argmax(dim=1) == labels).sum().item()
        return {'loss': sum_loss / size, 'eq': count_eq / size, 'size': size}

    @torch.inference_mode()
    def predict(self, dataset):
        self.model.eval()
        tensor = torch.linspace(0,
            len(dataset) - 1,
            steps=const.STEPS).long().tolist()
        stack = torch.stack([dataset[j][0] for j in tensor])
        max_list = self.model(stack).argmax(dim=1).tolist()
        return [
            [j, int(dataset[j][1]),
                int(max_list[i]),
                stack[i].squeeze(0).tolist()]
            for i, j in enumerate(tensor)
        ]


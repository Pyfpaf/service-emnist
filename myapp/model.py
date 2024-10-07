import os

import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim=784, output_dim=47):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.layer_2(x)
        return out


# class Model:
#     def __init__(self):
#         model_path = os.path.join('myapp', 'model.pth')
#         self.model = torch.load(model_path)
#         self.model.eval()
#         # your code here
#
    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание
        '''
        x = torch.from_numpy(x.reshape(-1, 28 * 28).astype(np.float32))

        with open('emnist-balanced-mapping.txt', 'r') as file:
            content = file.read()

        lines = content.strip().split('\n')
        labels_dict = {int(lines[i].split(' ')[0]): chr(int(lines[i].split(' ')[1])) for i in range(len(lines))}

        with torch.no_grad():
            pred = self.model(x)

        return labels_dict[pred.argmax().item()]




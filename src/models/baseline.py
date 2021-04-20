import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as M

class SmallNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, learning_rate=3e-4):
        super().__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        
        self.learning_rate = learning_rate

        self.criterion = nn.BCELoss()

    def forward(self, input_tensor):
        x = input_tensor.flatten(start_dim=1)
        hidden_state = F.relu(self.hidden(x))
        output_state = torch.sigmoid(self.linear(hidden_state))
        return output_state

    def _common_step(self, batch):
        x, y = batch
        y_hat = self(x).view(-1)
        loss = self.criterion(y_hat, y.float())
        acc = M.accuracy(y_hat, y)

        return loss, acc

    def training_step(self, batch, _):
        train_loss, train_acc = self._common_step(batch)

        self.log('train_loss', train_loss)
        self.log('train_acc', train_acc)

        return train_loss


    def validation_step(self, batch, _):
        val_loss, val_acc = self._common_step(batch)

        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)

    def test_step(self, batch, _):
        test_loss, test_acc = self._common_step(batch)

        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

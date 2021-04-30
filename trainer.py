import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

class Trainer:
    def __init__(self, network, train_dataloader, eval_dataloader, criterion, optimizer,
                 config, device):
        self.device = device
        self.config = config
        self.network = network
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self, epoch, total_epoch):
        self.network.train()
        running_loss = []
        accuracy = []
        for idx, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            output = self.network(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.criterion(output.float(), data.y.float())
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())
        print("[TRAIN] Epoch {}/{}, Accuracy is {}, Loss is {}".format(epoch, total_epoch, np.mean(accuracy),
                                                                   np.mean(running_loss)))
        return np.mean(accuracy), np.mean(running_loss)

    def eval_net(self):
        running_eval_loss = []
        self.network.eval()
        accuracy = []
        for idx, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            output = self.network(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = self.criterion(output.float(), data.y.float())
            # predictions = torch.argmax(output, dim=1)
            # accuracy.append(accuracy_score(label, predictions.cpu()))
            running_eval_loss.append(loss.item())
        print("[EVAL] Accuracy is {}, Loss is {}".format(np.mean(accuracy), np.mean(running_eval_loss)))
        return np.mean(accuracy), np.mean(running_eval_loss)

    def train(self):
        training_loss = []
        validation_loss = []

        training_accuracy = []
        validation_accuracy = []
        for i in range(1, self.config['epochs'] + 1):
            acc, loss = self.train_epoch(i, self.config['epochs'])
            training_loss.append(loss)
            training_accuracy.append(acc)

            acc, loss = self.eval_net()
            validation_loss.append(loss)
            validation_accuracy.append(acc)
        return training_accuracy, training_loss, validation_accuracy, validation_loss
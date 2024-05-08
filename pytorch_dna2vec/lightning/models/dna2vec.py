from typing import Any

import torch
import pytorch_lightning as L
import torchmetrics


class Dna2Vec(L.LightningModule):
    def __init__(self, vocabulary, embedding_dimension=32, device="cpu"):
        super(Dna2Vec, self).__init__()
        self.vocabulary = vocabulary
        self.vocabulary_size = self.vocabulary.__len__()
        self.embedding_dimension = embedding_dimension
        self.custom_device = device
        self.embedding = torch.nn.Embedding(self.vocabulary_size, embedding_dimension,
                                            padding_idx=self.vocabulary["pad"],  # implicit dependency on vocabulary padding
                                            device=self.custom_device)
        self.linear = torch.nn.Linear(embedding_dimension, self.vocabulary_size, device=self.custom_device)
        self.learning_rate = 0.01
        self.dense_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, fused=True)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.dense_optimizer, T_max=7, eta_min=.00001)

        self.scaler = torch.cuda.amp.GradScaler()
        self.criterion = torch.nn.CrossEntropyLoss()

        # self.accuracy = torchmetrics.classification.BinaryAccuracy()

    def forward(self, context):
        embeds = self.embedding(context).mean(dim=1)
        output = self.linear(embeds)
        return output

    def configure_optimizers(self):
        return self.dense_optimizer

    def training_step(self, batch, batch_idx):
        input, targets = batch
        input = input.to(device=self.custom_device)
        targets = targets.to(device=self.custom_device)
        self.dense_optimizer.zero_grad(set_to_none=True)

        # lengths should remain on cpu as all processing what needs lengths must be done on cpu
        with torch.autocast(device_type=self.custom_device, dtype=torch.float16):
            output = self(input)
            loss = self.criterion(output,
                             targets)
        # self.accuracy(output, targets)
        return loss

        # scaler.scale(loss).backward()
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad = param.grad.to_sparse()
        # scaler.step(dense_optimizer)
        # scaler.update()
        #
        # _, predicted = torch.max(output, 1)
        # correct = (predicted == targets).float().sum().item()
        # total_loss += loss.item()
        # batch_accuracy = correct / targets.size(0)
        # total_accuracy += batch_accuracy
        # batch_count += 1

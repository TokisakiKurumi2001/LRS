import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from LRS import LRSModel, LRSConfig, get_lr_linear_decay
from transformers import get_constant_schedule_with_warmup
import evaluate
import numpy as np
import re

class LitLRS(pl.LightningModule):
    def __init__(self, lr: float, num_keep_steps: int, num_training_steps: int):
        super(LitLRS, self).__init__()
        config = LRSConfig()
        self.model = LRSModel(config)
        self.lr = lr
        self.num_keep_steps = num_keep_steps
        self.num_training_steps = num_training_steps
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def training_step(self, batch, batch_idx):
        labels = batch.pop('label')
        o = self.model(**batch)

        loss = self.loss(o.view(-1, 2), labels.view(-1).long())

        self.log("train/loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': get_lr_linear_decay(optimizer, self.num_keep_steps, self.num_training_steps),
            'interval': 'step',
            'frequency': 1,
            'name': 'lr_monitor'
        }
        return [optimizer], [lr_scheduler]

from transformers import PreTrainedModel, XLMRobertaModel, M2M100ForConditionalGeneration
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Union, Optional, Tuple, Dict
from LRS import LRSConfig

class LRSPreTrainedModel(PreTrainedModel):
    config_class = LRSConfig
    base_model_prefix = "lrs"
    supports_gradient_checkpointing = True

class LRSModel(LRSPreTrainedModel):
    def __init__(self, config: LRSConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Linear(self.config.input_dim, self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, self.config.output_dim)

        self.post_init()

    def forward(self, sequence: Tensor) -> Tensor:
        # receive the output embedding of the encoder model
        x = self.embedding(sequence.float())
        x = self.out(x)

        return x
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple
import random
from copy import deepcopy

class LRSDataLoader:
    def __init__(self):
        data_dict = {'train': 'data/data.train.csv'}
        dataset = load_dataset('csv', data_files=data_dict)
        random.seed(42)
        self.dataset = dataset.map(
            self.__tokenize,
            remove_columns=dataset["train"].column_names
        )

    def __tokenize(self, examples):
        rt_dict = {}
        rt_dict['label'] = examples['label']
        rt_dict['sequence'] = [int(i) for i in examples['seq'].split(",")]
        return rt_dict

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples
        
        # convert list to tensor
        batch = {k: torch.tensor(v, dtype=torch.int32) for k, v in encoded_inputs.items()}

        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "test"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=1)
            )
        return res

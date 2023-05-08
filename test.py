from LRS import LRSDataLoader, LRSModel, LRSConfig
dataloader = LRSDataLoader()
[train_dataloader] = dataloader.get_dataloader(batch_size=2, types=['train'])
for batch in train_dataloader:
    print(batch)
    break

config = LRSConfig()
model = LRSModel(config)
labels = batch.pop('label')
out = model(**batch)
print(out)

import torch.nn as nn
l = nn.CrossEntropyLoss()
loss = l(out.view(-1, 2), labels.view(-1).long())
print(loss)
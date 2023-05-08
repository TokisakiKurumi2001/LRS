from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from LRS import LitLRS, LRSDataLoader
from math import ceil

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_lr_monitor")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    num_examples = 20000
    batch_size = 64
    epochs = 10
    hyperparameter = {
        "lr": 1e-3,
        "num_keep_steps": int(0.1 * ceil(num_examples / batch_size * epochs)),
        "num_training_steps": int(ceil(num_examples / batch_size * epochs))
    }
    lit_lrs = LitLRS(**hyperparameter)

    # dataloader
    lrs_dataloader = LRSDataLoader()
    [train_dataloader] = lrs_dataloader.get_dataloader(batch_size=batch_size, types=["train"])

    # train model
    trainer = pl.Trainer(max_epochs=epochs, devices=[1], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model=lit_lrs, train_dataloaders=train_dataloader)

    # save model & tokenizer
    lit_lrs.export_model('lrs_model/v1')

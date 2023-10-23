from typing import Union
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from nn.trainer.trainer_utils import prepare_datadict, create_default_loggers
from nn.trainer.trainer_constants import DEFAULT_BATCH_SIZE, DEFAULT_EARLY_STOPPING_PATIENCE, DEFAULT_EPOCHS


def train(model_class, data_args, train_params={}, 
          callbacks=None, 
          loggers: Union[None, Logger] = None,
          **kwargs):
    # Prepare datasets
    loaded_datasets = prepare_datadict(data_args)
    train_dataloader = DataLoader(loaded_datasets["train"], batch_size=train_params.get('batch_size', DEFAULT_BATCH_SIZE), shuffle=True)
    test_dataloader = DataLoader(loaded_datasets.get("test", loaded_datasets["validation"]), batch_size=train_params.get('batch_size', DEFAULT_BATCH_SIZE), shuffle=False)
    eval_dataloader = None if "test" not in loaded_datasets else DataLoader(loaded_datasets["test"], batch_size=train_params.get('batch_size', DEFAULT_BATCH_SIZE), shuffle=False)

    # Model, Optimizer, and Criterion Initialization
    model = model_class(**kwargs)
    if loggers is None:
        model_name = model.base_model_prefix or train_params.get('model_name', 'model')
        loggers = create_default_loggers(model_name)

    # Callbacks
    if callbacks is None:
        callbacks = []
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=train_params.get('early_stopping_patience', DEFAULT_EARLY_STOPPING_PATIENCE), 
                                   verbose=True, mode='min')
    callbacks.extend([checkpoint_callback, early_stopping])

    # Trainer Configuration
    gpus = train_params.get('gpus', 1)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        gpus = -1
        train_params["distributed_backend"] = "ddp"
        
    trainer = pl.Trainer(
        max_epochs=train_params.get('max_epochs', DEFAULT_EPOCHS),
        logger=loggers,
        callbacks=callbacks,
        gpus=gpus,
        auto_scale_batch_size='power' if gpus > 1 else None,
        **train_params,
    )
    
    if gpus > 1 or gpus == -1:
        trainer.tune(model, train_dataloader)

    trainer.fit(model, train_dataloader, test_dataloader)


    if eval_dataloader:
        trainer.test(test_dataloaders=eval_dataloader)
    
    return model, trainer



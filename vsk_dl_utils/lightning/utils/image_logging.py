import logging

import torch
from lightning import LightningModule

# Tensorboard
try:
    from lightning.pytorch.loggers import TensorBoardLogger

    tb_available = True
except ModuleNotFoundError:
    logging.warning("Tensorboard is not available")
    tb_available = False
    TensorBoardLogger = None

# WandB
try:
    import wandb
    from lightning.pytorch.loggers.wandb import WandbLogger

    wandb_available = True
except ModuleNotFoundError:
    logging.warning("WandB is not available")
    wandb_available = False
    WandbLogger = None

# Aim
try:
    from aim import Image as AimImage
    from aim.pytorch_lightning import AimLogger

    aim_available = True
except ModuleNotFoundError:
    logging.warning("Aim is not available")
    aim_available = False
    AimLogger = None


def log_pl_image(pl_module: LightningModule, image: torch.Tensor, step=None, name="train_image"):
    # Image must be in range 0..1
    if step is None:
        step = pl_module.global_step
    for logger in pl_module.loggers:
        print("Log image", pl_module.global_step)  # To avoid segmentation fault
        if tb_available and isinstance(logger, TensorBoardLogger):
            logger.experiment.add_image(name, image, step)
        elif wandb_available and isinstance(logger, WandbLogger):
            logger.experiment.log({name: [wandb.Image(image)]}, step=step)
        elif aim_available and isinstance(logger, AimLogger):
            aim_image = AimImage(image)
            logger.experiment.track(aim_image, name=name, step=step)

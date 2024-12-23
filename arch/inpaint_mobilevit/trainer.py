import os
from ...models.unext import UNet2DConditionModelControlNext
from diffusers import AutoencoderKL
import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

from ...dataset import collate_fn
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,DDIMScheduler)
from ...args import parse_args

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from copy import deepcopy
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import torch.nn.functional as F

if is_wandb_available():
    import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from .lightpipeline import MobileVitLightningPipe
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0.dev0")
from ...utils import get_dtype_training
logger = get_logger(__name__)
# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
from ...trainer import ModelTrainManager
from ...models.mobile_vit import CustomMobileViT
from ...dataset import AbstractDataset

class InpaintMobileVitManager(ModelTrainManager):
    def __init__(self, args):
        super().__init__(args)
        self.mobile_vit = None
        
    def init_unet(self):
        try:
            self.unet = UNet2DConditionModelControlNext.from_pretrained(
                self.args.pretrained_model_name_or_path,subfolder="unet", revision=self.args.revision, variant=self.args.variant
            ).to('cuda', dtype= self.dtype)
            logger.info("\033[92mSuccessfully initialized UNet model\033[0m")
        except Exception as e:
            logger.error(f"\033[91mFailed to initialize UNet model: {e}\033[0m")
            
    def init_mobile_vit(self):
        try:
            self.mobile_vit = CustomMobileViT(embedding_dim=640)
        except Exception as e:
            logger.error(f"\033[91mFailed to initialize MobileViT model: {e}\033[0m")

    def run_load_trainable_model(self):
        self.init_unet()
        self.init_mobile_vit()
        self.init_noise_scheduler()

        return self.unet, self.noise_scheduler, self.mobile_vit
    
def main():
    accelerator = Accelerator()

    args = parse_args()
    dtype = torch.bfloat16
    model_trainable = InpaintMobileVitManager(args)
            
    wandb_logger = WandbLogger(
        project="train inpainting",
        log_model=False)
    
    #Embedding data
    if args.save_embeddings_to_npz:
        vae, tokenizer, text_encoder  = model_trainable.run_load_embedding_model()
        
        lit_model = MobileVitLightningPipe(
        unet= None,
        mobilet_vit=None,
        vae= vae,
        text_encoder= text_encoder,
        tokenizer= tokenizer,
        mode= args.mode,
        noise_scheduler= None,
        args = args,   
    )
        dataset = AbstractDataset(
        data_path=args.dataset_path,
        input_type=args.input_type,
        dtype=dtype,
        image_size=(512,512)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
        lit_model.preprocess_data(dataloader= dataloader, weight_dtype= torch.bfloat16)
        del vae
        del tokenizer
        del text_encoder
        del lit_model
        del dataloader
        del dataset
        
    unet, noise_scheduler, mobile_vit_model = model_trainable.run_load_trainable_model()
    
    lit_model = MobileVitLightningPipe(
        unet= unet,
        mobilet_vit=mobile_vit_model,
        vae= None,
        text_encoder= None,
        tokenizer= None,
        mode= args.mode,
        noise_scheduler= noise_scheduler,
        args = args,   
    )
    
    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="model-{epoch:02d}-{train_loss:.4f}",
        save_top_k=5,
        mode="min",
        # every_n_epochs=199,  # Save every 5 epochs

    )
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    dataset = AbstractDataset(
        data_path=args.embedding_dir,
        input_type='embedding',
        dtype=dtype,
        image_size=(512,512)
        )

    trainer = Trainer(
            max_epochs=args.num_train_epochs,         
            accelerator= 'gpu',
            devices=1,
            callbacks=[checkpoint_callback, lr_monitor],
            logger=wandb_logger,
            log_every_n_steps=1,
            precision=16 if args.mixed_precision == "fp16" else 32,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            
        )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
        )
    # Train the model
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
    )

    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "final_checkpoint.ckpt")
    trainer.save_checkpoint(final_checkpoint_path)

    print(f"Training complete")


if __name__ == '__main__':
    main()

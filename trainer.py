import os
from .models.unext import UNet2DConditionModel
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
from .args import parse_args

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
from .lightningpipe import AbstractLightningPipe
from .pipeline import AbstractPipeline
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.29.0.dev0")
from .utils import get_dtype_training
logger = get_logger(__name__)
# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ModelTrainManager:
    def __init__(self, args):
        self.args = args
        self.unet = None
        self.controlnext = None
        self.vae = None
        self.noise_scheduler = None
        self.tokenizer = None
        self.text_encoder = None
        self.dtype = get_dtype_training(self.args.mixed_precision)

    def init_unet(self):
        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                self.args.pretrained_model_name_or_path,subfolder="unet", revision=self.args.revision, variant=self.args.variant
            ).to('cuda', dtype= self.dtype)
            logger.info("\033[92mSuccessfully initialized UNet model\033[0m")
        except Exception as e:
            logger.error(f"\033[91mFailed to initialize UNet model: {e}\033[0m")

    def init_vae(self):
        try:
            self.vae = AutoencoderKL.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=self.args.revision,
                variant=self.args.variant,
            ).to('cuda', dtype= self.dtype)
            logger.info("\033[92mSuccessfully initialized VAE model\033[0m")
        except Exception as e:
            logger.error(f"\033[91mFailed to initialize VAE model: {e}\033[0m")

    def init_tokenizer(self): 
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer", revision=self.args.revision, use_fast=False
            )
            logger.info("\033[92mSuccessfully initialized tokenizer\033[0m")
        except Exception as e:
            logger.error(f"\033[91mFailed to initialize tokenizer: {e}\033[0m")

    def init_noise_scheduler(self):
        try:
            self.noise_scheduler = DDIMScheduler.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type=self.args.prediction_type
            )
            logger.info("\033[92mSuccessfully initialized noise scheduler\033[0m")
        except Exception as e:
            logger.error(f"\033[91mFailed to initialize noise scheduler: {e}\033[0m")

    def init_text_encoder(self):
        try:
            text_encoder_config = PretrainedConfig.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=self.args.revision,
            )
            model_class = text_encoder_config.architectures[0]

            if model_class == "CLIPTextModel":
                from transformers import CLIPTextModel
                self.text_encoder = CLIPTextModel.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    subfolder="text_encoder",
                    revision=self.args.revision,
                ).to(device = 'cuda', dtype = self.dtype)
                logger.info("\033[92mSuccessfully initialized text encoder: CLIPTextModel\033[0m")
            else:
                raise ValueError(f"{model_class} is not supported.")
        except Exception as e:
            logger.error(f"\033[91mFailed to initialize text encoder: {e}\033[0m")

    def run_load_embedding_model(self):
        self.init_vae()
        self.init_tokenizer()
        self.init_text_encoder()

        return self.vae, self.tokenizer, self.text_encoder 

    def run_load_trainable_model(self):
        self.init_unet()
        self.init_noise_scheduler()

        return self.unet, self.noise_scheduler


class SimpleDataset_Embedding(torch.utils.data.Dataset):
    
    def __init__(self, num_samples=10):
        self.data = [
            {
                "latents_target": torch.randn(3, 512, 512, dtype=torch.bfloat16),
                "prompt": torch.randn(77, 768, dtype=torch.bfloat16),
            }
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SimpleDataset_Training(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [
        {"latent_target": torch.randn(4, 64, 64).to(dtype = torch.bfloat16),
         "encoder_hidden_states": torch.randn(77, 768).to(dtype = torch.bfloat16)}
            for _ in range(10)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_embedding(examples):
    latents_target = torch.stack([example["latent_target"] for example in examples]).to(memory_format=torch.contiguous_format)
    encoder_hidden_states = torch.stack([example["encoder_hidden_states"] for example in examples]).to(memory_format=torch.contiguous_format)

    return {
        "latent_target": latents_target,
        "encoder_hidden_states": encoder_hidden_states,
    }


def main():

    accelerator = Accelerator()

    args = parse_args()
    model_trainable = ModelTrainManager(args)
            
    wandb_logger = WandbLogger(
        project="train inpainting",
        log_model=False)

    #Embedding data
    if args.save_embeddings_to_npz:
        vae, tokenizer, text_encoder  = model_trainable.run_load_embedding_model()

        lit_model = AbstractLightningPipe(
        unet= None,
        vae= vae,
        text_encoder= text_encoder,
        tokenizer= tokenizer,
        mode= args.mode,
        noise_scheduler= None,
        args = args,

    )
        dataset = SimpleDataset_Embedding()
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
        )
        lit_model.preprocess_data(dataloader= dataloader, weight_dtype= torch.bfloat16)
        del vae
        del tokenizer
        del text_encoder
        del lit_model
        del dataloader
        del dataset

    unet, noise_scheduler = model_trainable.run_load_trainable_model()

    lit_model = AbstractLightningPipe(
        unet= unet,
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

    dataset = SimpleDataset_Training()
    # Trainer configuration
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
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_embedding,
        pin_memory=True,
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

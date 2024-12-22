import os
from ...models.unext import UNet2DConditionModel    #Using U-Net base model instead
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

# from ...dataset import collate_fn
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import  set_seed
from tqdm.auto import tqdm
from ...args import parse_args

from diffusers.utils import check_min_version, is_wandb_available
from copy import deepcopy
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch.nn.functional as F

if is_wandb_available():
    import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from .lightpipeline import CatVtonLightningPipe
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
from .dataset import CatVtonDataset
from .skip_connection import skip_encoder_hidden_state, SkipAttnProcessor

class InpaintCatVtonManager(ModelTrainManager):
    def __init__(self, args):
        super().__init__(args)
        
    def init_unet(self):
        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder = 'unet', revision=self.args.revision, variant=self.args.variant
            )
            skip_encoder_hidden_state(self.unet,cross_attn_cls= SkipAttnProcessor )
            logger.info("Successfully initialized UNet model")
        except Exception as e:
            logger.error(f"Failed to initialize UNet model: {e}")
            
    
def main():
    accelerator = Accelerator()

    args = parse_args()
    dtype = torch.bfloat16
    model_trainable = InpaintCatVtonManager(args)
            
    wandb_logger = WandbLogger(
        project="train inpainting",
        log_model=False)
    
    #Embedding data
    if args.save_embeddings_to_npz:
        vae, tokenizer, text_encoder  = model_trainable.run_load_embedding_model()
        
        lit_model = CatVtonLightningPipe(
        unet= None,
        vae= vae,
        text_encoder= text_encoder,
        tokenizer= tokenizer,
        mode= args.mode,
        noise_scheduler= None,
        args = args,   
    )
        dataset = CatVtonDataset(
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
        
    unet, noise_scheduler = model_trainable.run_load_trainable_model()

    lit_model = CatVtonLightningPipe(
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
    dataset = CatVtonDataset(
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
            precision='bf16-mixed',
            accumulate_grad_batches=args.gradient_accumulation_steps,
            
        )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        # collate_fn=collate_fn,
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
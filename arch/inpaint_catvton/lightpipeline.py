import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from ...lightningpipe import AbstractLightningPipe
from ...utils import compute_dream_and_update_latents_for_inpaint, get_dtype_training
from ...pipeline import AbstractPipeline
from torch.utils.checkpoint import checkpoint


# def print_vram_usage(tag=""):
#     """Hàm in mức sử dụng VRAM hiện tại."""
#     allocated = torch.cuda.memory_allocated() / 1024 ** 3  # Bộ nhớ đang sử dụng (GB)
#     reserved = torch.cuda.memory_reserved() / 1024 ** 3    # Bộ nhớ đã cấp phát (GB)
#     print(f"[{tag}] VRAM Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
class CatVtonLightningPipe(AbstractLightningPipe):
    
    def __init__(self, args, unet, vae, text_encoder, tokenizer, mode, noise_scheduler):
        super(CatVtonLightningPipe, self).__init__(unet=unet, vae=vae, text_encoder= text_encoder, tokenizer= tokenizer, mode= mode, noise_scheduler= noise_scheduler,args= args)
        self.abstract_pipeline = AbstractPipeline()
        
    
    def training_step(self, batch, batch_idx):

        # Define latent_concat dim
        concat_dim = -2 
        
        # Get data_embedding
        device= 'cuda'
        dtype = get_dtype_training(self.args.mixed_precision)
        latents_target = batch["latents_target"].to(dtype =dtype )
        latent_masked = batch['latents_masked'].to(dtype = dtype)
        mask_pixel_values = batch['mask_pixel_values'].to(dtype = dtype)
        fill_pixel_values = batch['fill_images'].to(dtype = dtype)
        # Create timesteps
        bsz = latents_target.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,)).to(device = device)
        # Concatenate conditional
        masked_latent_concat = torch.cat([latent_masked, fill_pixel_values], dim=concat_dim)
        # print("CHECK: ",masked_latent_concat.device)
        height_mask, width_mask = mask_pixel_values.shape[2], mask_pixel_values.shape[3]        
        mask_latent=  torch.nn.functional.interpolate(mask_pixel_values, size=( height_mask// 8, width_mask // 8))
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)

        # Padding latent target to fit shape masked_latent and mask
        latent_target_concat = torch.cat((latents_target, fill_pixel_values), dim=concat_dim).to(dtype= dtype)


        # Add noise to latents_target
        noise = torch.randn_like(latent_target_concat).to(dtype= dtype)

        noisy_latents_target = self.noise_scheduler.add_noise(latent_target_concat, noise, timesteps)
  
        inpainting_latent_model_input = torch.cat([noisy_latents_target, mask_latent_concat, masked_latent_concat], dim=1).to(dtype= dtype)

    # DREAM integration
        # print_vram_usage('before using dreambooth')

        self.unet.to(device = device , dtype = dtype)
        inpainting_latent_model_input, latents_target = compute_dream_and_update_latents_for_inpaint(
            unet=self.unet,
            noise_scheduler=self.noise_scheduler,
            timesteps=timesteps,
            noise=noise,
            noisy_latents=inpainting_latent_model_input,
            target=latent_target_concat,
            encoder_hidden_states=None,
            dream_detail_preservation=1.0,  # You can adjust this value
        )

        inpainting_latent_model_input.to(dtype= dtype)
        model_pred = self(
            noisy_latents= inpainting_latent_model_input.to(dtype= dtype),
            time_steps= timesteps.to('cuda'),
            encoder_hidden_states = None,
        ) 

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latent_target_concat, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = latent_target_concat
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
  
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(self.noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, 5.0 * torch.ones_like(timesteps)], dim=1).min(
            dim=1
        )[0]
        if self.noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True,on_epoch=True,logger=True, sync_dist=True)

        return loss
    
    def preprocess_data(self, dataloader, weight_dtype):
        npz_data = {} 
        metadata_records = []
        
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
            #encoder latent and   
        # for i, batch in tqdm(enumerate(islice(dataloader, 1)), total= 1, desc="Processing batches"): #test with data loader 1 items
            latents_target = self.vae.encode(batch["latents_target"].to(dtype=weight_dtype,device = 'cuda')).latent_dist.sample()
            latents_target = latents_target * self.vae.config.scaling_factor   
            
            latents_masked = self.vae.encode(batch["latents_masked"].to(dtype=weight_dtype,device = 'cuda')).latent_dist.sample()
            latents_masked = latents_masked * self.vae.config.scaling_factor   

            mask_pixel_values = batch['mask_pixel_values'].to(dtype=weight_dtype,device = 'cuda')
            
            fill_pixel_values = self.vae.encode(batch["fill_images"].to(dtype=weight_dtype,device = 'cuda')).latent_dist.sample()
            fill_pixel_values = fill_pixel_values * self.vae.config.scaling_factor   
            
            #build meta data
            npz_data[f"latents_target_{i}"] = latents_target.to(self.dtype).detach().cpu().numpy()
            npz_data[f"latents_masked_{i}"] = latents_masked.to(self.dtype).detach().cpu().numpy()
            npz_data[f"mask_pixel_values_{i}"] = mask_pixel_values.to(self.dtype).detach().cpu().numpy()
            npz_data[f"fill_images_{i}"] = fill_pixel_values.to(self.dtype).detach().cpu().numpy()

            metadata_records.append({
                "batch_index": i,
                "latents_target_key": f"latents_target_{i}",
                "latents_masked_key": f"latents_masked_{i}",
                "mask_pixel_values_key": f"mask_pixel_values_{i}",
                "fill_images_key": f"fill_images_{i}",
            })

        np.savez_compressed(os.path.join(self.args.embedding_dir, "embeddings_data.npz"), **npz_data)
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_parquet(os.path.join(self.args.embedding_dir, "metadata.parquet"))
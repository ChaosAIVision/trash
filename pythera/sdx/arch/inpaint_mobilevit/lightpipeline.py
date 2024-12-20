import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from .pipeline import  InpaintMobileVitPipeline
from ...lightningpipe import AbstractLightningPipe
from ...utils import compute_dream_and_update_latents_for_inpaint, get_dtype_training
from transformers import AutoImageProcessor, BatchFeature
from itertools import islice
from pipeline import InpaintMobileVitPipeline

class MobileVitLightningPipe(AbstractLightningPipe):
    
    def __init__(self, args, unet, vae, text_encoder, tokenizer, mode, noise_scheduler, mobilet_vit):
        super().__init__(unet= unet, vae= vae, text_encoder= text_encoder, tokenizer= tokenizer, mode= mode, args= args, noise_scheduler= noise_scheduler, mobile_vit= mobilet_vit)
        # self.save_hyperparameters()
        # self.args = args
        self.mobilet_vit = mobilet_vit
        self.abstract_pipeline = InpaintMobileVitPipeline()
    
    def foward(self, noise_latents, time_steps, encoder_hidden_states):
        return self.abstract_pipeline.forward_pipeline(noise_latents = noise_latents, time_steps = time_steps, encoder_hidden_states = encoder_hidden_states)

    def training_step(self, batch, batch_idx):
        
        # Prepare data
        dtype = get_dtype_training (self.args.mixed_precision)
        latents_target = batch['latents_target'].to(dtype)
        latents_masked = batch['latents_masked'].to(dtype)
        mask_pixel_values = batch['mask_pixel_values'].to(dtype)
        encoder_hidden_state = batch['encoder_hidden_state'].to(dtype)
        fill_images = batch['fill_images'].to(dtype)
        height_mask, width_mask = mask_pixel_values.shape[2], mask_pixel_values.shape[3]        
        mask_latents = torch.nn.functional.interpolate(mask_pixel_values, size=( height_mask// 8, width_mask // 8))
        # Create timesteps
        angles= torch.tensor([1]).to(device ='cuda', dtype= dtype)
        timesteps = self.abstract_pipeline._forward_create_timesteps(latents_target).to(device = 'cuda')
        
         # Add noise
        noisy_latents, noise = self.abstract_pipeline._forward_add_noise(latents_target, timesteps)
        
        inpainting_latent_model_input = torch.cat([noisy_latents, mask_latents, latents_masked], dim=1)
        
        input_mobile_vit = BatchFeature(data={'pixel_values': fill_images.to(dtype = dtype)})
        self.mobile_vit.to(dtype)
        with torch.no_grad():
            output_mobilevit = self.abstract_pipeline._forward_mobile_vit(inputs= input_mobile_vit, time_steps= timesteps, angles= angles)

       

        # use Dream method th enhance adapt Unet
        self.unet.to(dtype = dtype)
        inpainting_latent_model_input, latents_target== compute_dream_and_update_latents_for_inpaint(
            unet=self.unet,
            noise_scheduler=self.noise_scheduler,
            timesteps=timesteps,
            noise=noise,
            noisy_latents=inpainting_latent_model_input,
            target=latents_target,
            encoder_hidden_states=encoder_hidden_state,
            dream_detail_preservation=1.0,
            conditional_controls = output_mobilevit)
        
        model_pred = self(noisy_latents= inpainting_latent_model_input, time_steps = timesteps,angles = angles, encoder_hidden_states = encoder_hidden_state, fill_images = input_mobile_vit)
        
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents_target, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = latents_target
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
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
            fill_images = batch['fill_images'].to(dtype=weight_dtype,device = 'cuda')

            #encoder text
            encoder_hidden_state = torch.randn(1,77,768).to('cuda', dtype= weight_dtype)
            #build meta data
            npz_data[f"latents_target_{i}"] = latents_target.to(self.dtype).detach().cpu().numpy()
            npz_data[f"latents_masked_{i}"] = latents_masked.to(self.dtype).detach().cpu().numpy()
            npz_data[f"mask_pixel_values_{i}"] = mask_pixel_values.to(self.dtype).detach().cpu().numpy()
            npz_data[f"fill_images_values_{i}"] = fill_images.to(self.dtype).detach().cpu().numpy()
            npz_data[f"encoder_hidden_state_{i}"] = encoder_hidden_state.to(self.dtype).detach().cpu().numpy()

            metadata_records.append({
                "batch_index": i,
                "latents_target_key": f"latents_target_{i}",
                "latents_masked_key": f"latents_masked_{i}",
                "mask_pixel_values_key": f"mask_pixel_values_{i}",
                "fill_images_key": f"fill_images_values_{i}",
                "encoder_hidden_state_key": f"encoder_hidden_state_{i}",
            })

        np.savez_compressed(os.path.join(self.args.embedding_dir, "embeddings_data.npz"), **npz_data)
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_parquet(os.path.join(self.args.embedding_dir, "metadata.parquet"))
            

        

        
        
        
        
from lightning.pytorch import LightningModule
import torch
from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

class AbstractPipeline():
    def __init__(self, unet, vae, text_encoder, tokenizer, mode, noise_scheduler):
        super().__init__() 
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.mode = mode
        self.noise_scheduler = noise_scheduler

    def forward(self, **kwargs):

        if int(self.mode) == 0:  # Trainable U-Net
            noisy_latents = kwargs.get("noisy_latents", None)
            time_steps = kwargs.get("time_steps", None)
            encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
            model_pred = self._forward_unet(noisy_latents=noisy_latents,time_steps = time_steps, encoder_hidden_states= encoder_hidden_states )
            return model_pred
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _forward_unet(self, noisy_latents, time_steps, encoder_hidden_states, **kwargs):
        
        model_pred = self.unet(
            sample=noisy_latents,
            timestep=time_steps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        return model_pred

    def _forward_vae_encoder(self, pixel_values):
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def _forward_add_noise(self, latents_target, timesteps):
        noise = torch.randn_like(latents_target)
        noisy_latents_target = self.noise_scheduler.add_noise(latents_target, noise, timesteps)
        return noisy_latents_target, noise

    def _forward_create_timesteps(self, latents_target):
        bsz = latents_target.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,))
        return timesteps
import torch
from dataclasses import dataclass
from typing import Optional, Union
import torch
import torch.nn as nn

class AbstractPipeline():

    def forward_pipeline(self,unet,noisy_latents,time_steps,encoder_hidden_states, **kwargs):

        model_pred = self._forward_unet(unet, noisy_latents=noisy_latents,time_steps = time_steps, encoder_hidden_states= encoder_hidden_states )
        
        return model_pred
  

    def _forward_unet(self,unet, noisy_latents, time_steps, encoder_hidden_states, **kwargs):
        

        model_pred = unet(
            sample=noisy_latents,
            timestep=time_steps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        return model_pred

    def _forward_vae_encoder(self, vae, pixel_values):

        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        return latents

    def _forward_add_noise(self, noise_scheduler, latents_target, timesteps):

        noise = torch.randn_like(latents_target)
        noisy_latents_target = noise_scheduler.add_noise(latents_target, noise, timesteps)

        return noisy_latents_target, noise

    def _forward_create_timesteps(self, noise_scheduler, latents_target):

        bsz = latents_target.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))

        return timesteps
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_snr
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

class AbstractLightningPipe(LightningModule):
    def __init__(self, pipeline, args):
        super(AbstractLightningPipe, self).__init__()
        self.save_hyperparameters()
        self.pipeline = pipeline
        self.args = args

    def forward(self, noisy_latents, time_steps, encoder_hidden_states, **kwargs):
        return self.pipeline.forward(noisy_latents, time_steps, encoder_hidden_states, **kwargs)

    def training_step(self, batch, batch_idx):
        # Prepare data
        latent_target = batch['latent_target']
        encoder_hidden_states = batch["encoder_hidden_state"]

        # Create timesteps
        timesteps = self.pipeline._forward_create_timesteps(latent_target)

        # Add noise
        noisy_latents, noise = self.pipeline._forward_add_noise(latent_target, timesteps)

        # Forward pass through pipeline
        model_pred = self(
            noisy_latents=noisy_latents,
            time_steps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Calculate target based on noise_scheduler prediction type
        if self.pipeline.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.pipeline.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.pipeline.noise_scheduler.get_velocity(latent_target, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.pipeline.noise_scheduler.config.prediction_type}")

        # Calculate loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.pipeline.unet.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.adam_weight_decay
        )
        return optimizer
    
    def preprrocess_data(self, dataloader, weight_dtype):
        npz_data = {} 
        metadata_records = [] 
        
        for i, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader), desc="Processing batches"):
            #encoder latent and   
            latents_target = self.pipeline.vae.encode(batch["latents_target"].to(dtype=weight_dtype,device = 'cuda')).latent_dist.sample()
            latents_target = latents_target * self.vae.config.scaling_factor
            
            #encoder text
            input_text = (self.pipeline.tokenizer(batch['prompt'], max_length=self.tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids).to('cuda')
            encoder_hidden_state = self.text_encoder(input_text, return_dict=False)[0]  

            #build meta data
            npz_data[f"latents_target_{i}"] = latents_target.to(self.dtype).detach().cpu().numpy()
            npz_data[f"encoder_hidden_state_{i}"] = encoder_hidden_state.to(self.dtype).detach().cpu().numpy()

            metadata_records.append({
                "batch_index": i,
                "latents_target_key": f"latents_target_{i}",
                "encoder_hidden_state_key": f"encoder_hidden_state_{i}",
            })

        np.savez_compressed(os.path.join(self.args.embedding_dir, "embeddings_data.npz"), **npz_data)
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_parquet(os.path.join(self.embedding_dir, "metadata.parquet"))
            

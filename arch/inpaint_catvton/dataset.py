import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageChops
from ...utils import get_dtype_training
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pandas as pd
from diffusers.image_processor import VaeImageProcessor
from colorama import Fore, init
from ...dataset import TransformImage, AbstractDataset
#itnit colorama 
init(autoreset=True)

class CatVtonDataset(AbstractDataset):
    
    def __init__(self, data_path, input_type, dtype, image_size):
        super().__init__(data_path, input_type, dtype, image_size)
        
    
     
    def make_data(self, image_path, fill_image_path, bbox_string):
        
        pixel_values = Image.open(image_path).convert("RGB")
        bboxes_annotation = self.get_bboxes_annotations(bbox_string)
        mask_pixel_values = self.get_bbox_mask(pixel_values,bboxes_annotation)
        latents_masked = self.get_masked_image_bbox(pixel_values, bboxes_annotation)
        fill_images = self.get_fill_image_bboxes(pixel_values, bboxes_annotation)
        return {
            
            'latents_target': self.transform.vae_image_transform(image=pixel_values, height= self.image_size[0], width= self.image_size[1] ),
            'latents_masked': self.transform.vae_image_transform(image=latents_masked, height= self.image_size[0], width= self.image_size[1]),
            'fill_images': self.transform.vae_image_transform(image=fill_images, height= self.image_size[0], width= self.image_size[1] ),
            'mask_pixel_values': self.transform.vae_mask_transform(image=mask_pixel_values, height= self.image_size[0], width= self.image_size[1] )
              
             
        }
             
    def load_saved_embeddings(self, idx):
        """
        Dont need encoder hidden states training

    
        """
        
        npz_file = np.load(os.path.join( self.data, "embeddings_data.npz"))
        metadata_df = pd.read_parquet(os.path.join( self.data, "metadata.parquet"))

        if idx is not None:
            if idx < 0 or idx >= len(metadata_df):
                raise ValueError(f"Index {idx} out of range. Must be between 0 and {len(metadata_df) - 1}.")            
            row = metadata_df.iloc[idx]
            

            return {
                "latents_target": npz_file[row["latents_target_key"]],
                "latents_masked": npz_file[row["latents_masked_key"]],
                "mask_pixel_values": npz_file[row["mask_pixel_values_key"]],
                "fill_images": npz_file[row["fill_images_key"]],
                                
            }
   
   
    def __getitem__(self, index):
        
        """
        Remove encoder hidden states in pipeline
    
        """
        
        if self.input_type == 'csv':
            item = self.data.iloc[index]
            image_path = item['image_path']
            fill_image = None
            bbox_string = item['bbox']
            batch = self.make_data(image_path,fill_image, bbox_string)
            
            return batch
        
        if self.input_type == 'embedding':
            data= self.load_saved_embeddings(index)
            return {
                "latents_target": torch.tensor(data['latents_target']).to(dtype= self.dtype).squeeze(0),
                "latents_masked": torch.tensor(data['latents_masked']).to(dtype= self.dtype).squeeze(0),
                "mask_pixel_values": torch.tensor(data['mask_pixel_values']).to(dtype= self.dtype).squeeze(0),
                "fill_images": torch.tensor(data['fill_images']).to(dtype= self.dtype).squeeze(0)}
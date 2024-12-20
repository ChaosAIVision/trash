import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageChops
from .utils import get_dtype_training
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pandas as pd
from diffusers.image_processor import VaeImageProcessor
from colorama import Fore, init
#itnit colorama 
init(autoreset=True)




def collate_fn(examples):
    latents_target = torch.stack([example["latents_target"] for example in examples])
    latents_target = latents_target.to(memory_format=torch.contiguous_format)
    
    latents_masked = torch.stack([example['latents_masked'] for example in examples])
    latents_masked = latents_masked.to(memory_format=torch.contiguous_format)
 
    fill_pixel_values = torch.stack([example['fill_images'] for example in examples])
    fill_pixel_values = fill_pixel_values.to(memory_format=torch.contiguous_format)
    
    mask_pixel_values = torch.stack([example['mask_pixel_values'] for example in examples])
    mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format)
    
    encoder_hidden_state = torch.stack([example['encoder_hidden_state'] for example in examples])
    encoder_hidden_state = encoder_hidden_state.to(memory_format=torch.contiguous_format)
    
    # index_fill_image = torch.stack([example['index_fill_image'] for example in examples])
    # index_fill_image = index_fill_image.to(memory_format=torch.contiguous_format)
    
    return {
        "latents_target": latents_target,
        "latents_masked": latents_masked,
        'fill_images': fill_pixel_values,
        'mask_pixel_values': mask_pixel_values,
        'encoder_hidden_state': encoder_hidden_state,
        # 'index_fill_image': index_fill_image

    }


class TransformImage():
    
    def __init__(self):
        
        self.vae_image_processor = VaeImageProcessor(vae_scale_factor= 8)
        
        self.vae_mask_processor =  VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
        
        
        
    def vae_image_transform(self, image, height, width):
        
        return  self.vae_image_processor.preprocess(image,height, width)[0] 
          
    def vae_mask_transform(self, image, height, width):
        
        return self.vae_mask_processor.preprocess(image, height, width)[0]
    
    def pytorch_image_transform(self, image, height, width):
        
        return transforms.Compose([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    
    

class AbstractDataset(Dataset):
    def __init__(self, data_path , input_type, dtype, image_size):
        
        self.input_type = input_type
        
        # input data csv and target is make embedding data
        if self.input_type == 'csv':
            self.data = pd.read_csv(data_path)
        if self.input_type == 'embedding':
            self.data = Path(data_path)
            metadata_file = self.data / 'metadata.parquet'
            if not metadata_file.exists():
                raise FileNotFoundError(Fore.RED + f'metadata.parquet not in {self.data}')
            self.metadata = pd.read_parquet(metadata_file)
        self.image_size = image_size

        self.dtype = get_dtype_training(dtype)
        self.transform = TransformImage()
        
    def __len__(self):
        
        if self.input_type == 'csv':
            return len(self.data)
        if self.input_type == 'embedding':
            return len(self.metadata)
        
    def string_to_list(self, str_annotation):
        """
        Converts a string of annotations to a list of integers.

        Args:
            str_annotation: 'x1,y1,x2,y2' to list [x1,y1,x2,y2,x3, y3]
        """
        
        if isinstance(str_annotation, float):
            str_annotation = str(int(str_annotation))
        
        return list(map(int,str_annotation.split(',')))
            
    def get_bboxes_annotations(self, bbox_string):
        """
        Convert bbox string to boundingbox list

        Args:
            bbox_string: 'x1,y1,x2,y2' to list
        """
        
        return self.string_to_list(bbox_string)
    
    def get_bbox_mask(self, image, bbox):
        
        """
        Creates a black-and-white mask where the region inside the bbox is white,
        and the region outside is black.

        Args:
            image : PIL image
            bbox : List Bbox
            
        """
        mask = Image.new('L', image.size, color = 0)
        x1, y1, x2, y2 = bbox
        for x in range(x1, x2):
            for y in range(y1, y2):
                mask.putpixel((x,y), 225)
        return mask
    
    def get_masked_image_bbox(self, image ,bbox):
        """

        Args:
            image : PIL image target
            bbox : list bbox
            
        """
        image_array = np.array(image)  # Shape: (H, W, C)
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_array.shape[1], x2), min(image_array.shape[0], y2)

        black_overlay = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)  

        masked_image_array = image_array.copy()
        masked_image_array[y1:y2, x1:x2] = black_overlay

        masked_image = Image.fromarray(masked_image_array.astype(np.uint8))


        return masked_image
    
    def get_fill_image_bboxes(self, images, bbox_annotations):
        
        """
        Creates an image with the area inside bbox filled with white.
        """
        xmin, ymin, xmax, ymax = bbox_annotations
        cropped_image = images.crop((xmin, ymin, xmax, ymax))
          
        return cropped_image
        
    
    def make_data(self, image_path, fill_image_path, bbox_string):
        
        pixel_values = Image.open(image_path).convert("RGB")
        bboxes_annotation = self.get_bboxes_annotations(bbox_string)
        mask_pixel_values = self.get_bbox_mask(pixel_values,bboxes_annotation)
        latents_masked = self.get_masked_image_bbox(pixel_values, bboxes_annotation)
        fill_images = Image.open(fill_image_path).convert("RGB")
        
        return {
            
            'latents_target': self.transform.vae_image_transform(image=pixel_values, height= self.image_size[0], width= self.image_size[1] ),
            'latents_masked': self.transform.vae_image_transform(image=latents_masked, height= self.image_size[0], width= self.image_size[1]),
            'fill_images': self.transform.vae_image_transform(image=fill_images, height= self.image_size[0], width= self.image_size[1] ),
            'mask_pixel_values': self.transform.vae_mask_transform(image=mask_pixel_values, height= self.image_size[0], width= self.image_size[1] )
              
             
        }
        
    def load_saved_embeddings(self, idx):
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
                "encoder_hidden_state": npz_file[row["encoder_hidden_state_key"]],
                "fill_images": npz_file[row["fill_images_key"]],
                                
            }
            
    def __getitem__(self, index):
        
        # Prepare for read data from csv and make embedding data
        if self.input_type == 'csv':
            item = self.data.iloc[index]
            image_path = item['image_path']
            fill_image = item['fill_image_path']
            bbox_string = item['bbox']
            batch = self.make_data(image_path,fill_image, bbox_string)
            
            return batch
        
        if self.input_type == 'embedding':
            data= self.load_saved_embeddings(index)
            return {
                "latents_target": torch.tensor(data['latents_target']).to(dtype= self.dtype).squeeze(0),
                "latents_masked": torch.tensor(data['latents_masked']).to(dtype= self.dtype).squeeze(0),
                "mask_pixel_values": torch.tensor(data['mask_pixel_values']).to(dtype= self.dtype).squeeze(0),
                "encoder_hidden_state": torch.tensor(data['encoder_hidden_state']).to(dtype= self.dtype).squeeze(0),
                "fill_images": torch.tensor(data['fill_images']).to(dtype= self.dtype).squeeze(0)}
            


            

            
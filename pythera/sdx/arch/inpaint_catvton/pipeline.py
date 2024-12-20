from pythera.sdx.pipeline import AbstractPipeline

class InpaintCatVTonPipeline(AbstractPipeline):
    def __init__(self, unet, vae, text_encoder, tokenizer, mode, noise_scheduler, args):
        super().__init__(unet= unet, vae =vae, text_encoder= text_encoder, tokenizer= tokenizer, mode= mode, noise_scheduler = noise_scheduler, args = args)
    
    
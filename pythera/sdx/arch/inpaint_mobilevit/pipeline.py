from pythera.sdx.pipeline import AbstractPipeline

class InpaintMobileVitPipeline(AbstractPipeline):
  def __init__(self, unet, vae, text_encoder, tokenizer, mode, noise_scheduler,args,  mobile_vit):
    # super().__init__(unet= unet, vae = vae, text_encoder= text_encoder, tokenizer= tokenizer, mode= mode, noise_scheduler= noise_scheduler, args= args)
    # self.mobile_vit = mobile_vit
    super.__init__()
    

  def _forward_mobile_vit(self, mobile_vit, inputs, time_steps, angles):
    return mobile_vit(inputs, time_steps, angles)
  
  def _forward_unet(self, unet, noisy_latents, time_steps, encoder_hidden_states, **kwargs):
    ''''
    conditional_controls: output of mobile_vit send into u_net
    
    '''

    conditional_controls = kwargs.get('conditional_controls', None)
    model_pred = unet(
        sample=noisy_latents,
        timestep=time_steps,
        encoder_hidden_states=encoder_hidden_states,
        conditional_controls=conditional_controls,
        return_dict=False,
    )[0]
    return model_pred

  def forward_pipeline(self, **kwargs):
    '''
    fill_images: image fill into background
    
    '''
    noisy_latents = kwargs.get('noisy_latents', None)
    time_steps = kwargs.get('time_steps', None)
    angles = kwargs.get('angles', None)
    encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
    mobile_vit = kwargs.get("mobile_vit")
    unet = kwargs.get("unet")

    fill_images = kwargs.get('fill_images', None)
    if fill_images == None:
      raise ValueError('Fill images is None!')

    output_mobile_vit = self._forward_mobile_vit(mobile_vit, inputs=fill_images, time_steps=time_steps, angles=angles)
    output_unet = self._forward_unet(unet, noisy_latents=noisy_latents, time_steps=time_steps, encoder_hidden_states=encoder_hidden_states, conditional_controls=output_mobile_vit)
    return output_unet
    


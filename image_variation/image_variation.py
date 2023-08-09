
import open_clip
import torch
import types
import random
import os.path as osp
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, PNDMScheduler,UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.training_utils import set_seed


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def read_img(img):
    if isinstance(img, str):
        if osp.exists(img):
            image = Image.open(img)
        else:
            image = download_image(img)
        return image
    elif isinstance(img, Image.Image):
        return img
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img)
    else:
        raise
    
def get_model(ckpt_dir):
    def encode_image(self, image, ):
        self.visual.output_tokens=True
        pooled, tokens=self.visual(image)
        return pooled[:,None]
    
    scheduler = UniPCMultistepScheduler(beta_start=0.00085,beta_end=0.012,beta_schedule='scaled_linear')
    vae = AutoencoderKL.from_pretrained(ckpt_dir, subfolder='vae')
    vae.eval()
    unet = UNet2DConditionModel.from_pretrained(ckpt_dir, subfolder='unet')
    unet.eval()
    cond_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-H-14',
        pretrained=osp.join(ckpt_dir,'CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin')                                                                                      )
    cond_model.encode_image=types.MethodType(encode_image, cond_model)
    cond_model.eval()
    return scheduler, vae, unet, cond_model, preprocess_val

class ImageVariation():
    def __init__(self, ckpt_dir, norm_file, device=torch.device('cuda:0'), dtype=torch.float16) -> None:
        self.device=device
        self.dtype=dtype
        scheduler, vae, unet, cond_model, preprocess_val=get_model(ckpt_dir)
        self.scheduler: PNDMScheduler=scheduler
        self.vae=vae.to(self.device,self.dtype)
        self.unet=unet.to(self.device,self.dtype)
        self.cond_model=cond_model.to(self.device,self.dtype)
        self.preprocess=preprocess_val
        self.norm_file=norm_file
    
    def noise_image_embeddings(self,image_embeds,noise_level,generator=None):
        noise = randn_tensor(image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype)
        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)
        
        meanstd=torch.from_numpy(np.load(self.norm_file)[None]).to(device=image_embeds.device, dtype=image_embeds.dtype)
        mean,std=torch.chunk(meanstd,2,dim=1)
        #scale
        image_embeds-=mean
        image_embeds/=std
        
        image_embeds=self.scheduler.add_noise(image_embeds,noise,noise_level)
        
        #unscale
        image_embeds*=std
        image_embeds+=mean
    
        return image_embeds
    
    @torch.no_grad()
    def __call__(self,
                 image,
                 num_inference_steps=20,
                 guidance_scale=5.0,
                 num_images_per_prompt=1,
                 seed=None,
                 height=512,
                 width=512,
                 noise_level: int=0,
                 ):
        if seed is None:
            seed=random.randint(0,2**32-1)
        set_seed(seed)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        mask=torch.ones(2,1,height//8,width//8,device=self.device,dtype=self.dtype)
        masked_img=torch.zeros(2,3,height,width,device=self.device,dtype=self.dtype)
        masked_image_latents = self.vae.encode(masked_img).latent_dist.sample()
        masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        
        mask=mask.repeat(num_images_per_prompt,1,1,1)
        masked_image_latents=masked_image_latents.repeat(num_images_per_prompt,1,1,1)
         
        latents = randn_tensor((num_images_per_prompt,4,height//8,width//8),device=self.device, generator=None,dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        cond_image=image
        clip_img=self.preprocess(read_img(cond_image).convert('RGB')).unsqueeze(0)
        cond_embedding=self.cond_model.encode_image(clip_img.to(self.device,self.dtype)).to(self.dtype)
        cond_embedding=cond_embedding.repeat(num_images_per_prompt,1,1)
        if noise_level>0:
            cond_embedding=self.noise_image_embeddings(cond_embedding,noise_level)
        cond_embedding=torch.cat([cond_embedding*0,cond_embedding])
        
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            
            noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=cond_embedding,cross_attention_kwargs={}).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **{}).prev_sample
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        imgs=[Image.fromarray(img) for img in image]
        return imgs

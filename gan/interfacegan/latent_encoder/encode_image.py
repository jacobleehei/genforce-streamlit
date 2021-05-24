import argparse
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
import os

from gan.interfacegan.latent_encoder.InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from gan.interfacegan.latent_encoder.models.latent_optimizer import LatentOptimizer
from gan.interfacegan.latent_encoder.models.image_to_latent import ImageToLatent
from gan.interfacegan.latent_encoder.models.losses import LatentLoss
from gan.interfacegan.latent_encoder.utilities.hooks import GeneratedImageHook
from gan.interfacegan.latent_encoder.utilities.files import validate_path
from gan.interfacegan.latent_encoder.utilities.images import save_image
from gan.interfacegan.latent_encoder.utilities.images import load_images

def optimize_latents(input_image, optimize_iterations):
    
    if input_image is None: return
    print("Optimizing Latents.")
    synthesizer = StyleGANGenerator("stylegan_ffhq").model.synthesis
    latent_optimizer = LatentOptimizer(synthesizer, 12)

    # Optimize only the dlatents.
    for param in latent_optimizer.parameters():
        param.requires_grad_(False)

    generated_image_hook = GeneratedImageHook(latent_optimizer.post_synthesis_processing, 1)

    reference_image = load_images([input_image])
    reference_image = torch.from_numpy(reference_image).cuda()
    reference_image = latent_optimizer.vgg_processing(reference_image)
    reference_features = latent_optimizer.vgg16(reference_image).detach()
    reference_image = reference_image.detach()
    latents_to_be_optimized = torch.zeros((1,18,512)).cuda().requires_grad_(True)
    
    criterion = LatentLoss()
    optimizer = torch.optim.SGD([latents_to_be_optimized], lr=1)

    progress_bar = tqdm(range(optimize_iterations))
    for step in progress_bar:
        optimizer.zero_grad()
        generated_image_features = latent_optimizer(latents_to_be_optimized)
        
        loss = criterion(generated_image_features, reference_features)
        loss.backward()
        loss = loss.item()

        optimizer.step()
        progress_bar.set_description("Step: {}, Loss: {}".format(step, loss))

    optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
    img = save_image(optimized_dlatents)
    return img, optimized_dlatents

def encode_image(optimize_iterations):
    
    image_path = os.path.abspath('img/interface_output/aligned_file/temp_01.png')
    dlatent_path = os.path.abspath('img/interface_output/optimized_file/upload')
    optimized_file_path = os.path.abspath('img/interface_output/optimized_file/upload.png')

    img = optimize_latents(image_path, dlatent_path, optimize_iterations, optimized_file_path)
    return(img)




    
    



import os
from tqdm import tqdm
import numpy as np
import torch
import cv2
import io
import bz2
import requests 
import dlib
from PIL import Image
import scipy.ndimage
import gc
import streamlit as st

from gan.interfacegan.InterFaceGAN.models.model_settings import MODEL_POOL
from gan.interfacegan.InterFaceGAN.models.pggan_generator import PGGANGenerator
from gan.interfacegan.InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from gan.interfacegan.latent_encoder.models.latent_optimizer import LatentOptimizer
from gan.interfacegan.latent_encoder.models.image_to_latent import ImageToLatent
from gan.interfacegan.latent_encoder.models.losses import LatentLoss
from gan.interfacegan.latent_encoder.utilities.hooks import GeneratedImageHook
from gan.interfacegan.latent_encoder.utilities.files import validate_path
from gan.interfacegan.latent_encoder.utilities.images import save_image
from gan.interfacegan.latent_encoder.utilities.images import load_images

class itfgan_webObject:

    def __init__(self):
        self.model = {
            'stylegan_ffhq': None,
            'stylegan_celebahq': None,
            'pggan_celebahq': None
        }
        self.synthesizer = None

    def build_generatorS(self):
        self.cleanCache()
        self.model['stylegan_ffhq']     = StyleGANGenerator('stylegan_ffhq')
        self.model['stylegan_celebahq'] = StyleGANGenerator('stylegan_celebahq')
        self.synthesizer = StyleGANGenerator("stylegan_ffhq").model.synthesis

    def build_generatorP(self):
        self.cleanCache()
        self.model['pggan_celebahq'] = PGGANGenerator('pggan_celebahq')
    
    def cleanCache(self):
        torch.cuda.empty_cache()

    def randomSamplig(self, model_name, latentSpaceType, num):
        self.cleanCache()
        for name in self.model:
            if model_name is name: generator = self.model[name] 

        if latentSpaceType == 'W': kwargs = {'latent_space_type': 'W'}
        else: kwargs = {}
        codes = generator.easy_sample(num)
        if latentSpaceType == 'W':
              codes = torch.from_numpy(codes).type(torch.FloatTensor).to(generator.run_device)
              codes = generator.get_value(generator.model.mapping(codes))

        origin_image = generator.easy_synthesize(codes, **kwargs)['image']

        return codes, origin_image


    def manipulate(self, 
        latentCode, model_name, latentSpaceType, 
        age, eyeglasses, gender, pose, smile, 
        check_if_upload = False
    ):
      self.cleanCache()    
      for name in self.model:
          if model_name is name: model = self.model[name]

      default_control_features = ['age','eyeglasses','gender','pose','smile']
      boundaries = {}
      for i, attr_name in enumerate(default_control_features):
          boundary_name = f'{model_name}_{attr_name}'
          if model_name == 'stylegan_ffhq' or model_name == 'stylegan_celebahq' :
            if latentSpaceType == 'W' or latentSpaceType == 'WP': 
              w_Sboundary = os.path.abspath(f'gan/interfacegan/InterFaceGAN/boundaries/{boundary_name}_w_boundary.npy')
              boundaries[attr_name] = np.load(w_Sboundary)
            else: 
              Sboundary = os.path.abspath(f'gan/interfacegan/InterFaceGAN/boundaries/{boundary_name}_boundary.npy')
              boundaries[attr_name] = np.load(Sboundary)
          else:
              Gboundary = os.path.abspath(f'gan/interfacegan/InterFaceGAN/boundaries/{boundary_name}_boundary.npy')
              boundaries[attr_name] = np.load(Gboundary)

      if latentSpaceType == 'W': kwargs = {'latent_space_type': 'W'}
      else: kwargs = {}

      if check_if_upload is True: kwargs = {'latent_space_type': 'WP'}

      latentCode = model.preprocess(latentCode, **kwargs)

      
      new_codes = latentCode.copy()
      for i, attr_name in enumerate(default_control_features):
          new_codes += boundaries[attr_name] * eval(attr_name)
      
      newImage = model.easy_synthesize(new_codes, **kwargs)['image']
      return newImage

    def optimize_latents(self, input_image, optimize_iterations):

            if input_image is None: return
            latent_optimizer = LatentOptimizer(self.synthesizer, 12)

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
            
            with st.spinner('inverting source image...⏳'):
                stprogress_bar = st.progress(0)
                progress_bar = tqdm(range(optimize_iterations))
                for step in progress_bar:
                    stprogress_bar.progress(step/optimize_iterations)
                    
                    optimizer.zero_grad()
                    generated_image_features = latent_optimizer(latents_to_be_optimized)
                    loss = criterion(generated_image_features, reference_features)
                    loss.backward()
                    loss = loss.item()

                    optimizer.step()
                    progress_bar.set_description("Step: {}, Loss: {}".format(step, loss))
            stprogress_bar.empty()
            optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
            return optimized_dlatents

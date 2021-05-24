import streamlit as st
import io, IPython.display, cv2, PIL.Image, os, gc, torch, time, sys
import numpy as np
from tqdm import tqdm

if sys.platform == 'darwin':
    sys.path.append('utils/ganFunction')
    import idinvertHandler as idv
else:
    import utils.ganFunction.idinvertHandler as idv

from gan.higan.models.helper import build_generator
from gan.higan.utils.logger import setup_logger
from gan.higan.utils.editor import get_layerwise_manipulation_strength
from gan.higan.utils.editor import manipulate


w1k_code = np.load(os.path.abspath('utils/ganFunction/order_w.npy'))

class higan_webOject:
 
    def __init__(self):
          self.inverter = None
          self.generator = None
          self.model = {
            'stylegan_bedroom'       : None,
            'stylegan_livingroom'    : None,
            'stylegan2_church'       : None,
            'stylegan_churchoutdoor' : None,
            'stylegan_tower'         : None,
            'stylegan_kitchen'       : None,
            'stylegan_bridge'        : None,
    }


    def cleanCache(self):
        gc.collect()
        torch.cuda.empty_cache()


    def build_generator(self, model_name):
          self.cleanCache()
          if self.model[model_name] is None:
              self.model[model_name] = build_model(model_name)

    def build_inverter(self):
          self.cleanCache()
          if self.inverter is None:
            self.inverter = idv.build_inverter('styleganinv_tower256')
            self.generator = idv.build_generator('styleganinv_tower256')

    def invert(self, input_image):
          code, _ = idv.invert(self.inverter, input_image)
          new_images = self.generator.easy_synthesize(code, latent_space_type='wp')['image']
          return new_images, code

    def randomImage(self, num_samples, noise_seed, model_name):  
      generator = self.model[model_name]

      if model_name == 'stylegan2_church':
            indoor_latent_codes = sample_codes_church(generator, num_samples, noise_seed)
      else:
            indoor_latent_codes = sample_codes(generator, num_samples, noise_seed, w1k_code=w1k_code)

      synthesis_kwargs = {'latent_space_type': 'wp'}
      images = generator.easy_synthesize(indoor_latent_codes, **synthesis_kwargs)['image']
      return images, indoor_latent_codes


    def manipulate(self, attribute_name, indoor_model_name, distance, indoor_latent_codes):
          
      generator = self.model[indoor_model_name]

      path = f'gan/higan/boundaries/{indoor_model_name}/{attribute_name}_boundary.npy'
      try:
        boundary_file = np.load(path, allow_pickle=True).item()
        boundary = boundary_file['boundary']
        manipulate_layers = boundary_file['meta_data']['manipulate_layers']
      except ValueError:
        boundary = np.load(path)
        if attribute_name == 'view':
          manipulate_layers = '0-4'
        else:
          manipulate_layers = '6-11'


      if attribute_name == 'view' or indoor_model_name == 'stylegan2_church':
        strength = [1.0 for _ in range(generator.num_layers)]
      else:
        strength = get_layerwise_manipulation_strength(
          generator.num_layers, generator.truncation_psi, generator.truncation_layers)

      indoor_codes = manipulate(latent_codes=indoor_latent_codes,
                          boundary=boundary,
                          start_distance=0,
                          end_distance=distance,
                          step=2,
                          layerwise_manipulation=True,
                          num_layers=generator.num_layers,
                          manipulate_layers=manipulate_layers,
                          is_code_layerwise=True,
                          is_boundary_layerwise=False,
                          layerwise_manipulation_strength=strength)

      images = generator.easy_synthesize(indoor_codes[:, 1], latent_space_type='wp')['image']
      return images




def build_model(model_name, logger=None):
  """Builds the generator by model name."""
  model = build_generator(model_name, logger=logger)
  return model

def sample_codes(model, num, seed=0, w1k_code=None):
  """Samples latent codes randomly."""
  np.random.seed(seed)
  if w1k_code is None:
    # codes = generator.easy_sample(num)
    latent_codes = model.easy_sample(num=num, latent_space_type='w')
  else:
    latent_codes = w1k_code[np.random.randint(0, w1k_code.shape[0], num)] 
  latent_codes = model.easy_synthesize(latent_codes=latent_codes,
                                       latent_space_type='w',
                                       generate_style=False,
                                       generate_image=False)['wp']
  return latent_codes

def sample_codes_church(model, num, seed=0):
  """Samples latent codes randomly."""
  np.random.seed(seed)
  #codes = generator.easy_sample(num)
  latent_codes = model.easy_sample(num=num, latent_space_type='w')
  latent_codes = model.easy_synthesize(latent_codes=latent_codes,
                                       latent_space_type='w',
                                       generate_style=False,
                                       generate_image=False)['wp']
  return latent_codes
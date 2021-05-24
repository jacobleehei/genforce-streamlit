# python3.7
"""Edits latent codes with respect to given boundary.

Basically, this file takes latent codes and a semantic boundary as inputs, and
then shows how the image synthesis will change if the latent codes is moved
towards the given boundary.

NOTE: If you want to use W or W+ space of StyleGAN, please do not randomly
sample the latent code, since neither W nor W+ space is subject to Gaussian
distribution. Instead, please use `generate_data.py` to get the latent vectors
from W or W+ space first, and then use `--input_latent_codes_path` option to
pass in the latent vectors.
"""

import os.path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import PIL
import io

from gan.interfacegan.latent_encoder.InterFaceGAN.models.model_settings import MODEL_POOL
from gan.interfacegan.latent_encoder.InterFaceGAN.models.pggan_generator import PGGANGenerator
from gan.interfacegan.latent_encoder.InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from gan.interfacegan.latent_encoder.InterFaceGAN.utils.logger import setup_logger
from gan.interfacegan.latent_encoder.InterFaceGAN.utils.manipulator import linear_interpolate


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Edit image synthesis with given semantic boundary.')
  parser.add_argument('-m', '--model_name', type=str, required=True,
                      choices=list(MODEL_POOL),
                      help='Name of the model for generation. (required)')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-b', '--boundary_path', type=str, required=True,
                      help='Path to the semantic boundary. (required)')
  parser.add_argument('-i', '--input_latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
  parser.add_argument('-n', '--num', type=int, default=1,
                      help='Number of images for editing. This field will be '
                           'ignored if `input_latent_codes_path` is specified. '
                           '(default: 1)')
  parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                      choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                      help='Latent space used in Style GAN. (default: `Z`)')
  parser.add_argument('--start_distance', type=float, default=-3.0,
                      help='Start point for manipulation in latent space. '
                           '(default: -3.0)')
  parser.add_argument('--end_distance', type=float, default=3.0,
                      help='End point for manipulation in latent space. '
                           '(default: 3.0)')
  parser.add_argument('--steps', type=int, default=10,
                      help='Number of steps for image editing. (default: 10)')

  return parser.parse_args()


def interfacegan(model_name, input_latent_codes_path, latent_space_type, age, eyeglasses, gender, pose, smile):
  """Main function."""
  output_dir = os.path.abspath('img/interface_output/random_sampling/')


  default_control_features = ['age','eyeglasses','gender','pose','smile']
  boundaries = {}
  for i, attr_name in enumerate(default_control_features):
      boundary_name = f'{model_name}_{attr_name}'

      if model_name == 'stylegan_ffhq' or model_name == 'stylegan_celebahq' :
        if latent_space_type == 'W' or latent_space_type == 'WP':
          file_name_w = os.path.abspath(f'gan/interfacegan/latent_encoder/InterFaceGAN/boundaries/{boundary_name}_w_boundary.npy')
          boundaries[attr_name] = np.load(file_name_w)
        else:
          file_name = os.path.abspath(f'gan/interfacegan/latent_encoder/InterFaceGAN/boundaries/{boundary_name}_boundary.npy')
          boundaries[attr_name] = np.load(file_name)
      else:
          file_name = os.path.abspath(f'gan/interfacegan/latent_encoder/InterFaceGAN/boundaries/{boundary_name}_boundary.npy')
          boundaries[attr_name] = np.load(file_name)

  gan_type = MODEL_POOL[model_name]['gan_type']
  if gan_type == 'pggan':
    model = PGGANGenerator(model_name)
    kwargs = {}
  elif gan_type == 'stylegan':
    model = StyleGANGenerator(model_name)
    kwargs = {'latent_space_type': latent_space_type}
  else:
    raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')


  if os.path.isfile(input_latent_codes_path):
    latent_codes = np.load(input_latent_codes_path)
    latent_codes = model.preprocess(latent_codes, **kwargs)
    new_codes = latent_codes.copy()
    for i, attr_name in enumerate(default_control_features):
      new_codes += boundaries[attr_name] * eval(attr_name)

    new_images = model.easy_synthesize(new_codes, **kwargs)['image']
    return new_images
  else:
    latent_codes = model.easy_sample(1, **kwargs)
    np.save(os.path.join(output_dir, 'random_latent_codes.npy'), latent_codes)
    return new_images
    

def imshow(images, col, viz_size=1024):
  """Shows images in one figure."""
  num, height, width, channels = images.shape
  assert num % col == 0
  row = num // col

  fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

  for idx, image in enumerate(images):
    i, j = divmod(idx, col)
    y = i * viz_size
    x = j * viz_size
    if height != viz_size or width != viz_size:
      image = cv2.resize(image, (viz_size, viz_size))
    fused_image[y:y + viz_size, x:x + viz_size] = image

  fused_image = np.asarray(fused_image, dtype=np.uint8)
  data = io.BytesIO()
  PIL.Image.fromarray(fused_image).save(data, 'jpeg')
  im_data = data.getvalue()
  #disp = IPython.display.display(IPython.display.Image(im_data))
  return im_data
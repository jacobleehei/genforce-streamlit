import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import cv2

from gan.idinvert.utils.inverter import StyleGANInverter
from gan.idinvert.utils.logger import setup_logger
from gan.idinvert.utils.visualizer import HtmlPageVisualizer
from gan.idinvert.utils.visualizer import load_image, resize_image


def diffuse(target_img, context_img, crop_size, iterations):
  """Main function."""
  torch.cuda.empty_cache()
  model_name = 'styleganinv_ffhq256'

  os.environ["CUDA_VISIBLE_DEVICES"] = '0'


  inverter = StyleGANInverter(
      model_name,
      learning_rate=0.01,
      iteration=iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=5e-5,
      regularization_loss_weight=0.0,
      logger=None)
  image_size = inverter.G.resolution


  save_interval = iterations // 1
  headers = ['Target Image', 'Context Image', 'Stitched Image',
             'Encoder Output']
  for step in range(1, iterations + 1):
    if step == iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')


  latent_codes = []
  for target_idx in tqdm(range(1), desc='Target ID', leave=False):
    target_image = resize_image(target_img,
                                (image_size, image_size))
    for context_batch_idx in tqdm(range(0, 1, 4),
                            desc='Context ID', leave=False):
      context_images = []
      for it in range(1):
        context_idx = context_batch_idx + it
        if context_idx >= 1:
          continue
        row_idx = target_idx * 1 + context_idx
        context_image = resize_image(context_img,
                                     (image_size, image_size))
        context_images.append(context_image)

      code, viz_results = inverter.easy_diffuse(target=target_image,
                                                context=np.asarray(context_images),
                                                center_x=125,
                                                center_y=145,
                                                crop_x=crop_size,
                                                crop_y=crop_size,
                                                num_viz=5)
      for key, values in viz_results.items():
        context_idx = context_batch_idx + key
        row_idx = target_idx * 1 + context_idx
        for viz_idx, viz_img in enumerate(values):
          cropped_img = viz_img
      latent_codes.append(code)

  # Save results.
  np.save(f'img/indomain_output/result/inverted_codes.npy',
          np.concatenate(latent_codes, axis=0))
  cv2.imwrite('img/indomain_output/result/result.png',cropped_img)
  return cropped_img


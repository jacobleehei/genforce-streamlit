import os
from tqdm import tqdm
import numpy as np
from keras.utils import get_file
import io
import bz2
import requests
import dlib
import numpy as np
from PIL import Image
import IPython.display
import scipy.ndimage

from gan.idinvert.models.helper import build_generator
from gan.idinvert.utils.editor import interpolate
from gan.idinvert.utils.visualizer import load_image
from gan.idinvert.utils.inverter import StyleGANInverter

LANDMARK_MODEL_NAME = 'shape_predictor_68_face_landmarks.dat'
LANDMARK_MODEL_PATH = os.path.abspath('gan.idinvert/models/pretrain/shape_predictor_68_face_landmarks.dat')
LANDMARK_MODEL_URL = f'http://dlib.net/files/{LANDMARK_MODEL_NAME}.bz2'

def idInvert(src_img):

  def invert(inverter, image):
    """Inverts an image."""
    latent_code, reconstruction = inverter.easy_invert(image, num_viz=1)
    return latent_code, reconstruction

  model_name = 'styleganinv_ffhq256'
  inverted_code_dir = 'inverted_codes'
  print('Building inverter')
  inverter = build_inverter(model_name=model_name)
  print('Building generator')
  generator = get_generator(model_name)
  
  mani_image = align(inverter, src_img)
  if mani_image.shape[2] == 4:
    mani_image = mani_image[:, :, :3]


  latent_code_path = os.path.abspath('img/indomain_output/manipulation/latent.npy')
  latent_code, _ = invert(inverter, mani_image)
  np.save(latent_code_path, latent_code)
  

def manipulation_edit(age, gender, pose, eyeglasses, expression):
  
  latent_code_path = os.path.abspath('img/indomain_output/manipulation/latent.npy')
  model_name = 'styleganinv_ffhq256'
  generator = get_generator(model_name)
  latent_code = np.load(latent_code_path)

  new_codes = latent_code.copy()
  boundaries = {}
  ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'expression']
  for attr in ATTRS:
    boundary_path = os.path.join(os.path.abspath('src/utils/InDomainGAN/boundaries'), 
                                'stylegan_ffhq256', attr + '.npy')
    boundary_file = np.load(boundary_path, allow_pickle=True)[()]
    boundary = boundary_file['boundary']
    manipulate_layers = boundary_file['meta_data']['manipulate_layers']
    boundaries[attr] = []
    boundaries[attr].append(boundary)
    boundaries[attr].append(manipulate_layers)

  for i, attr_name in enumerate(ATTRS):
    manipulate_layers = boundaries[attr_name][1]
    new_codes[:, manipulate_layers, :] += boundaries[attr_name][0][:, manipulate_layers, :] * eval(attr_name)

  new_images = generator.easy_synthesize(new_codes, **{'latent_space_type': 'wp'})['image']
  return new_images


def interpolate(src_img, target_img, step, model):
  """Main function."""

  """add check if gpu available later"""
  os.environ["CUDA_VISIBLE_DEVICES"] = '0'
  
  inverted_code_dir = 'inverted_codes'
  os.makedirs(inverted_code_dir, exist_ok=True)

  def linear_interpolate(src_code, dst_code, step=5):
    assert (len(src_code.shape) == 2 and len(dst_code.shape) == 2 and
            src_code.shape[0] == 1 and dst_code.shape[0] == 1 and
            src_code.shape[1] == dst_code.shape[1])

    linspace = np.linspace(0.0, 1.0, step)[:, np.newaxis].astype(np.float32)
    return src_code + linspace * (dst_code - src_code)

  print('Building inverter')
  inverter = build_inverter(model)
  output_dir = os.path.abspath('img/indomain_output/interpolate/result')

  generator = build_generator(model)

  src_image = align(inverter, src_img)
  if src_image.shape[2] == 4:
      src_image = src_img[:, :, :3]
  src_code_path = os.path.join(inverted_code_dir, 'src.npy')
  

  dst_image = align(inverter, target_img)
  if dst_image.shape[2] == 4:
     dst_image = dst_image[:, :, :3]
  dst_code_path = os.path.join(inverted_code_dir, 'dst.npy')

  src_code, _ = invert(inverter, src_image)
  np.save(os.path.abspath('img/indomain_output/interpolate/context_temp/src.npy'), src_code)

  dst_code, _ = invert(inverter, dst_image)
  np.save(os.path.abspath('img/indomain_output/interpolate/target_temp/dst.npy'), dst_code)


  src_code = np.load(os.path.abspath('img/indomain_output/interpolate/context_temp/src.npy'))
  dst_code = np.load(os.path.abspath('img/indomain_output/interpolate/target_temp/dst.npy'))

  print('Start interpolation.')
  step = step + 2
  viz_size = 256
  
  inter_images = []
  inter_images.insert(0, dst_image)
  inter_images.insert(-1, src_image)

  inter_codes = linear_interpolate(np.reshape(src_code, [1, -1]),
                                  np.reshape(dst_code, [1, -1]),
                                  step=step)
  inter_codes = np.reshape(inter_codes, [-1, inverter.G.num_layers, inverter.G.w_space_dim])
  inter_imgs = generator.easy_synthesize(inter_codes, **{'latent_space_type': 'wp'})['image']

  for ind in range(inter_imgs.shape[0]):
    inter_images.insert(ind+1, inter_imgs[ind])

  inter_images = np.asarray(inter_images)

  return inter_imgs


def interfaceGAN_invert(src_img):
  def invert(inverter, image):
    """Inverts an image."""
    latent_code, reconstruction = inverter.easy_invert(image, num_viz=1)
    return latent_code, reconstruction

  inverter = build_inverter('styleganinv_ffhq256')
  src_image = align(inverter, src_img)
  if src_image.shape[2] == 4:
      src_image = src_img[:, :, :3]

  src_code, _ = invert(inverter, src_image)
  np.save(os.path.abspath('img/interface_output/optimized_file/upload.npy'), src_code)



class FaceLandmarkDetector(object):
  """Class of face landmark detector."""

  def __init__(self, align_size=256, enable_padding=True):
    """Initializes face detector and landmark detector.

  Args:
    align_size: Size of the aligned face if performing face alignment.
    (default: 1024)
    enable_padding: Whether to enable padding for face alignment (default:
    True)
  """
    # Download models if needed.
    if not os.path.exists(LANDMARK_MODEL_PATH):
      data = requests.get(LANDMARK_MODEL_URL)
      data_decompressed = bz2.decompress(data.content)
      with open(LANDMARK_MODEL_PATH, 'wb') as f:
        f.write(data_decompressed)

    self.face_detector = dlib.get_frontal_face_detector()
    self.landmark_detector = dlib.shape_predictor(LANDMARK_MODEL_PATH)
    self.align_size = align_size
    self.enable_padding = enable_padding

  def detect(self, image_path):
    results = []

    # image_ = np.array(image)
    images = dlib.load_rgb_image(image_path)
    bboxes = self.face_detector(images, 1)
    # Landmark detection
    for bbox in bboxes:
      landmarks = []
      for point in self.landmark_detector(images, bbox).parts():
        landmarks.append((point.x, point.y))
      results.append({
          'image_path': image_path,
          'bbox': (bbox.left(), bbox.top(), bbox.right(), bbox.bottom()),
          'landmarks': landmarks,
      })
    return results

  def align(self, face_info):
    """Aligns face based on landmark detection.

  The face alignment process is borrowed from
  https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py,
  which only supports aligning faces to square size.

  Args:
    face_info: Face information, which is the element of the list returned by
    `self.detect()`.

  Returns:
    A `np.ndarray`, containing the aligned result. It is with `RGB` channel
    order.
  """
    img = Image.open(face_info['image_path'])

    landmarks = np.array(face_info['landmarks'])
    eye_left = np.mean(landmarks[36: 42], axis=0)
    eye_right = np.mean(landmarks[42: 48], axis=0)
    eye_middle = (eye_left + eye_right) / 2
    eye_to_eye = eye_right - eye_left
    mouth_middle = (landmarks[48] + landmarks[54]) / 2
    eye_to_mouth = mouth_middle - eye_middle

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_middle + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / self.align_size * 0.5))
    if shrink > 1:
      rsize = (int(np.rint(float(img.size[0]) / shrink)),
               int(np.rint(float(img.size[1]) / shrink)))
      img = img.resize(rsize, Image.ANTIALIAS)
      quad /= shrink
      qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
      img = img.crop(crop)
      quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0),
           max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if self.enable_padding and max(pad) > border - 4:
      pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
      img = np.pad(np.float32(img),
                   ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                   'reflect')
      h, w, _ = img.shape
      y, x, _ = np.ogrid[:h, :w, :1]
      mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                         np.float32(w - 1 - x) / pad[2]),
                        1.0 - np.minimum(np.float32(y) / pad[1],
                                         np.float32(h - 1 - y) / pad[3]))
      blur = qsize * 0.02
      blurred_image = scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
      img += blurred_image * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
      img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
      img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
      quad += pad[:2]

    # Transform.
    img = img.transform((self.align_size * 4, self.align_size * 4), Image.QUAD,
                        (quad + 0.5).flatten(), Image.BILINEAR)
    img = img.resize((self.align_size, self.align_size), Image.ANTIALIAS)

    return np.array(img)


def align_face(image_path, align_size=256):
  """Aligns a given face."""
  model = FaceLandmarkDetector(align_size)
  face_infos = model.detect(image_path)
  face_infos = face_infos[0]
  img = model.align(face_infos)
  return img


def build_inverter(model_name, iteration=100, regularization_loss_weight=2):
  """Builds inverter"""
  inverter = StyleGANInverter(
      model_name,
      learning_rate=0.01,
      iteration=iteration,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=5e-5,
      regularization_loss_weight=regularization_loss_weight)
  return inverter


def get_generator(model_name):
  """Gets model by name"""
  return build_generator(model_name)


def align(inverter, image_path):
  """Aligns an unloaded image."""
  aligned_image = align_face(image_path,
                             align_size=inverter.G.resolution)
  return aligned_image



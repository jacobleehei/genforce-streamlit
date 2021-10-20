import os

import numpy as np
import torch

from gan.interfacegan.models.pggan_generator import PGGANGenerator
from gan.interfacegan.models.stylegan_generator import \
    StyleGANGenerator


class gan:

    def __init__(self):
        self.model = {'stylegan_ffhq': None}
        self.synthesizer = None

    def build(self):        
        if self.model['stylegan_ffhq'] == None:
            self.model = {'stylegan_ffhq': StyleGANGenerator('stylegan_ffhq')}
            self.synthesizer = self.model['stylegan_ffhq'].model.synthesis

    def randomSamplig(self, model_name, latentSpaceType, num):
        for name in self.model:
            if model_name is name:
                generator = self.model[name]

        if latentSpaceType == 'W':
            kwargs = {'latent_space_type': 'W'}
        else:
            kwargs = {}
        codes = generator.easy_sample(num)
        if latentSpaceType == 'W':
            codes = torch.from_numpy(codes).type(
                torch.FloatTensor).to(generator.run_device)
            codes = generator.get_value(generator.model.mapping(codes))

        origin_image = generator.easy_synthesize(codes, **kwargs)['image']

        return codes, origin_image

    def manipulate(self,
                   latentCode, model_name, latentSpaceType,
                   age, eyeglasses, gender, pose, smile,
                   check_if_upload=False
                   ):
        for name in self.model:
            if model_name is name:
                model = self.model[name]

        default_control_features = [
            'age', 'eyeglasses', 'gender', 'pose', 'smile']
        boundaries = {}
        for attr_name in default_control_features:
            boundary_name = f'{model_name}_{attr_name}'
            if model_name == 'stylegan_ffhq' or model_name == 'stylegan_celebahq':
                if latentSpaceType == 'W' or latentSpaceType == 'WP':
                    w_Sboundary = os.path.abspath(
                        f'gan/interfacegan/boundaries/{boundary_name}_w_boundary.npy')
                    boundaries[attr_name] = np.load(w_Sboundary)
                else:
                    Sboundary = os.path.abspath(
                        f'gan/interfacegan/boundaries/{boundary_name}_boundary.npy')
                    boundaries[attr_name] = np.load(Sboundary)
            else:
                Gboundary = os.path.abspath(
                    f'gan/interfacegan/boundaries/{boundary_name}_boundary.npy')
                boundaries[attr_name] = np.load(Gboundary)

        if latentSpaceType == 'W':
            kwargs = {'latent_space_type': 'W'}
        else:
            kwargs = {}

        if check_if_upload is True:
            kwargs = {'latent_space_type': 'WP'}

        latentCode = model.preprocess(latentCode, **kwargs)

        new_codes = latentCode.copy()
        for attr_name in default_control_features:
            new_codes += boundaries[attr_name] * eval(attr_name)

        newImage = model.easy_synthesize(new_codes, **kwargs)['image']
        return newImage

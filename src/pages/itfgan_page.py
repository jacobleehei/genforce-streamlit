import io
import os
import sys
import time
from tkinter import filedialog

import numpy as np
import streamlit as st
from PIL import Image

if sys.platform == 'darwin':
    sys.path.append('utils')
    import ganFunction.interfaceganHandler as itf
    from gan.interfacegan.face_align.align_images import align_image
    from webFunction import image_cropping, open_gif
else:
    import utils.ganFunction.interfaceganHandler as itf
    from gan.interfacegan.face_align.align_images import align_image
    from utils.webFunction import image_cropping, open_gif

itfTool = itf.itfgan_webObject()

def write():

    itfNode = [
        '📖 Overview',
        '✨ Upload an image',
        '✨ Randomly generate',
    ]

    st.title('InterFaceGAN')
    st.subheader('by Yujun Shen,  Jinjin Gu,  Xiaoou Tang,  Bolei Zhou1 ([@GenForce\_](https://genforce.github.io/))')

    st.markdown(
        "[![Paper(CVPR)](https://img.shields.io/badge/Paper(CVPR)-green?logo=xda-developers&logoColor=white)](https://arxiv.org/pdf/1907.10786.pdf)  &nbsp;&nbsp;"
        "[![Paper(TPAMI)](https://img.shields.io/badge/Paper(TPAMI)-green?logo=xda-developers&logoColor=white)](https://arxiv.org/pdf/2005.09635.pdf)  &nbsp;&nbsp;"
        "[![Colab](https://img.shields.io/badge/Colab-F9AB00?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/genforce/interfacegan/blob/master/docs/InterFaceGAN.ipynb) &nbsp;&nbsp;"
        "[![GitHub](https://img.shields.io/github/stars/genforce/interfacegan?style=social&label=Code&maxAge=2592000)](https://github.com/genforce/interfacegan)&nbsp;&nbsp;"      
        )

    if itfTool.model['stylegan_ffhq'] is None: build_itfTool()
    operation = st.selectbox('Please choose the operation you want',list(itfNode), index = 0)
    writePageNode(operation)


def writePageNode(operation):
    if operation == '📖 Overview':
        st.markdown("""
        
        Latent code for well-trained generative models, such as PGGAN and StyleGAN, actually learns a disentangled representation after some linear transformations. 
        Based on our analysis, InterFaceGAN as a simple and general technique is proposed for semantic face editing in latent space. 
        InterFaceGAN manage to control the pose as well as other facial attributes, such as gender, age, eyeglasses. 
        More importantly, InterFaceGAN are able to correct the artifacts made by GANs.
        
        ## Example: Manipulate the attributes with PGGAN
        """, unsafe_allow_html=True) 
        with st.spinner('Loading...☕'):
            open_gif(f'img/web/itfpage/example.gif')
            st.markdown('<br>', unsafe_allow_html=True)
            st.subheader('🎈Check more results in the following video.')
            st.video('https://youtu.be/uoftpl3Bj6w')

    if operation == '✨ Upload an image'  : upImageNode()
    if operation == '✨ Randomly generate': randomNode()


def upImageNode():

    Top,_ = st.beta_columns((1,0.000001))
    A1, A2 = st.beta_columns((2,1))

    Top.subheader('Step 1: Upload and Invert an Image')
    uploadImg = A1.file_uploader(label="Upload your image!", type=['jpg', 'png', 'jpeg'],accept_multiple_files=False)    
    quality = 100*(A1.slider('Invert quality (high qulity will lead to longer process time)', 1, 10, 5, 1))

    if uploadImg: 
        A2.image(uploadImg, 'uploaded photo', use_column_width= True)

    with st.sidebar:
        st.subheader('👱🏼 Feature')
        age = st.sidebar.slider('age', -3.0, 3.0, 0.0, 0.5)
        gender = st.sidebar.slider('gender', -3.0, 3.0, 0.0, 0.5)
        pose = st.sidebar.slider('pose', -3.0, 3.0, 0.0, 0.5)
        smile = st.sidebar.slider('smile', -3.0, 3.0, 0.0, 0.5)
        eyeglasses = st.sidebar.slider('eyeglasses', -3.0, 3.0, 0.0, 0.5)


    @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
    def invert():
        if uploadImg: 
            with st.spinner('Aligning image...⏳'):
                try: aligned, need_optimize = align_image(uploadImg)
                except: 
                    try: aligned, need_optimize = align_image(Image.open(uploadImg).transpose(Image.ROTATE_270))
                    except:
                        st.warning('Something going wrong of the input image☹️')
            return aligned, itfTool.optimize_latents(need_optimize, quality)
        else: return None, None

    aligned_image, latent = invert()

    if uploadImg:
        st.subheader('Step 2: Play with the features in the sidebar')
        with st.spinner('processing latent code...⏳'):
            newImage = itfTool.manipulate(
                    latent,'stylegan_ffhq', 'WP', age, eyeglasses, gender, pose, smile, True
                )  
        B1, B2 = st.beta_columns((1,1))
        B1.image(aligned_image, use_column_width = True)
        B2.image(newImage, use_column_width = True)
    

def randomNode():
    model_list = {
        'click here': '',
        'stylegan_ffhq': ['Z','W','WP'],
        'stylegan_celebahq': ['Z','W','WP'],
        'pggan_celebahq': ['Z']
    }

    st.subheader('Step 1: Set up the parameters')
    A1, A2 = st.beta_columns((1,1))
    model = A1.selectbox('Model', list(model_list.keys()))
    latentSpaceType = A2.selectbox('lantent space type', list(model_list[model]))
    
    numSamples = A1.slider('num of samples', 1, 5, 1, 1)
    seed = A2.slider('random seed', 1, 100, 25, 1)

    if sys.platform != 'darwin':
        @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
        def random(seed):
            if model_list[model] is not '':  
                with st.spinner('Generating samples...⏳'): 
                        return itfTool.randomSamplig(model, latentSpaceType, numSamples)
            else: return None, None
    else:
        def random(seed):
            if model_list[model] is not '':  
                with st.spinner('Generating samples...⏳'): 
                        return itfTool.randomSamplig(model, latentSpaceType, numSamples)
            else: return None, None
    


    latent, origin_image = random(seed)

    if model_list[model] is not '':
        with st.sidebar:
            st.subheader('👱🏼 Feature')
            age = st.slider('age', -3.0, 3.0, 0.0, 0.5)
            gender = st.slider('gender', -3.0, 3.0, 0.0, 0.5)
            pose = st.slider('pose', -3.0, 3.0, 0.0, 0.5)
            smile = st.slider('smile', -3.0, 3.0, 0.0, 0.5)
            eyeglasses = st.slider('eyeglasses', -3.0, 3.0, 0.0, 0.5)

    if sys.platform != 'darwin':
        @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
        def manipulate():
            return itfTool.manipulate(latent, model, latentSpaceType, age, eyeglasses, gender, pose, smile)
    else:
        def manipulate():
            return itfTool.manipulate(latent, model, latentSpaceType, age, eyeglasses, gender, pose, smile)
    


    with st.spinner('Loading samples...⏳'):
        newImage = manipulate()

    st.subheader('Step 2: Play with the features in the sidebar')
    B1, B2 = st.beta_columns((1,1))
    
    for i, o in zip(origin_image, newImage): 
            B1.image(i, 'Input', use_column_width=True)
            B2.image(o, 'Output', use_column_width=True)


def build_itfTool():
    progressbar = st.progress(1)

    with st.spinner('⏳ ...building generator for StyleGAN'):
        itfTool.build_generatorS()
        progressbar.progress(50)

    with st.spinner('⏳ ...building generator for PGGAN'):
        itfTool.build_generatorP()
        progressbar.progress(100)

    time.sleep(0.5)
    progressbar.empty()




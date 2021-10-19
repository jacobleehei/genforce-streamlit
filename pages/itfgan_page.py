import time

import streamlit as st

import utils.gan_handler.itfgan as itf
from utils.helper import open_gif

itfTool = itf.itfgan_webObject()


def write():

    itfNode = [
        '📖 Overview',
        '✨ Randomly generate',
        # '✨ Upload an image',
    ]

    st.title('InterFaceGAN')
    st.subheader(
        'by Yujun Shen,  Jinjin Gu,  Xiaoou Tang,  Bolei Zhou1 ([@GenForce\_](https://genforce.github.io/))')

    st.markdown(
        "[![Paper(CVPR)](https://img.shields.io/badge/Paper(CVPR)-green?logo=xda-developers&logoColor=white)](https://arxiv.org/pdf/1907.10786.pdf)  &nbsp;&nbsp;"
        "[![Paper(TPAMI)](https://img.shields.io/badge/Paper(TPAMI)-green?logo=xda-developers&logoColor=white)](https://arxiv.org/pdf/2005.09635.pdf)  &nbsp;&nbsp;"
        "[![Colab](https://img.shields.io/badge/Colab-F9AB00?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/genforce/interfacegan/blob/master/docs/InterFaceGAN.ipynb) &nbsp;&nbsp;"
        "[![GitHub](https://img.shields.io/github/stars/genforce/interfacegan?style=social&label=Code&maxAge=2592000)](https://github.com/genforce/interfacegan)&nbsp;&nbsp;"
    )

    operation = st.selectbox(
        'Please choose the operation you want', list(itfNode), index=0)
    write_page_node(operation)


def write_page_node(operation):
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

    if operation == '✨ Randomly generate':
        random_image_edit()


def random_image_edit():
    model_list = {
        'stylegan_ffhq': ['Z', 'W', 'WP'],
        'click here': '',
    }

    st.subheader('Step 1: Set up the parameters')
    A1, A2 = st.beta_columns((1, 1))
    model = A1.selectbox('Model', list(model_list.keys()))
    latentSpaceType = A2.selectbox(
        'lantent space type', list(model_list[model]))

    seed = st.slider('random seed', 1, 100, 25, 1)

    @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
    def random(seed):
        if model_list[model] is not '':
            with st.spinner('Generating samples...⏳'):
                return itfTool.randomSamplig(model, latentSpaceType, 1)
        else:
            return None, None

    latent, origin_image = random(seed)
    newImage = origin_image
    if model_list[model] is not '':
        with st.sidebar.form(key='feature'):
            st.subheader('👱🏼 Feature')
            submit = st.form_submit_button(label='submit🎉')
            age = st.slider('age', -3.0, 3.0, 0.0, 0.5)
            gender = st.slider('gender', -3.0, 3.0, 0.0, 0.5)
            pose = st.slider('pose', -3.0, 3.0, 0.0, 0.5)
            smile = st.slider('smile', -3.0, 3.0, 0.0, 0.5)
            eyeglasses = st.slider('eyeglasses', -3.0, 3.0, 0.0, 0.5)

            if submit:
                @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
                def manipulate():
                    return itfTool.manipulate(latent, model, latentSpaceType, age, eyeglasses, gender, pose, smile)

                with st.spinner('Loading samples...⏳'):
                    newImage = manipulate()

        st.subheader('Step 2: Play with the features in the sidebar')
        B1, B2 = st.beta_columns((1, 1))
        
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

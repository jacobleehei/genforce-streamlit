import streamlit as st

import utils.gan_handler.higan as higan
from utils.helper import download_button

hiTool = higan.higan_webOject()

model = {
    'click here': [''],
    'stylegan_bedroom': ['view', 'wood', 'cluttered_space', 'indoor_lighting', 'scary', 'glossy', 'dirt', 'carpet'],
    'stylegan_livingroom': ['carpet', 'cluttered_space', 'dirt', 'glossy', 'indoor_lighting', 'wood'],
    'stylegan_tower': ['brick', 'clouds', 'foliage', 'grass', 'sunny', 'trees', 'vegetation', 'vertical_components'],
}


def write():

    hiNode = [
        '📖 Overview',
        '✨ Do the magic of hierarchy images',
    ]

    st.title('HiGAN')
    st.subheader(
        'by Ceyuan Yang,  Yujun Shen,  Bolei Zhou([@GenForce\_](https://genforce.github.io/))')

    st.markdown(
        "[![Paper](https://img.shields.io/badge/Paper-green?logo=xda-developers&logoColor=white)](https://arxiv.org/pdf/1911.09267.pdf)  &nbsp;&nbsp;"
        "[![Colab](https://img.shields.io/badge/Colab-F9AB00?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/genforce/idinvert_pytorch/blob/master/docs/Idinvert.ipynb) &nbsp;&nbsp;"
        "[![GitHub](https://img.shields.io/github/stars/genforce/higan?style=social&label=Code&maxAge=2592000)](https://github.com/genforce/higan)&nbsp;&nbsp;"
    )

    operation = st.selectbox(
        'Please choose the operation you want', list(hiNode), index=0)
    write_node(operation)


def write_node(operation):

    if operation == '📖 Overview':
        st.markdown("""

        Highly-structured semantic hierarchy emerges from the generative representations as the variation factors for synthesizing scenes.
        By probing the layer-wise representations with a broad set of visual concepts at different abstraction levels,
        HiGAN is able to quantify the causality between the activations and the semantics occurring in the output image.

        ## Manipulation results
        Identifying such a set of manipulatable latent variation factors facilitates semantic scene manipulation.
        """, unsafe_allow_html=True)
        with st.spinner('Loading...☕'):
            st.subheader(
                'Check more results of various scenes in the following video.')
            st.video('https://youtu.be/X5yWu2Jwjpg')

    if operation == '✨ Do the magic of hierarchy images':
        rand_image_editing()


def rand_image_editing():

    st.subheader('Step1: Choose a generative model')
    selected_model = st.selectbox(
        'Choose your model to Generate random image!', list(model.keys()))

    if selected_model != 'click here':
        with st.spinner('Loading ' + selected_model + ' model...⏳'):
            hiTool.build_generator(selected_model)

    st.subheader('Step2: Play with the parameters🔨')
    selected_feature = st.multiselect('select a feature to edit!', list(
        model[selected_model]), default=list(model[selected_model][:3]))
    distance = st.slider('distance', -5.0, 5.0, 0.0, 1.0)
    if selected_model == 'stylegan2_church':
        distance *= 2
    noise_seed = st.slider('noise_seed', 0, 100, 10, 1)

    @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
    def random():
        with st.spinner('Generating samples...⏳'):
            return hiTool.randomImage(1, noise_seed, selected_model)

    @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
    def manipulate():
        output_image = {}
        with st.spinner('Loading samples...⏳'):
            for i in range(len(model[selected_model])):
                output_image[model[selected_model][i]] = (hiTool.manipulate(
                    model[selected_model][i], selected_model, distance, latent))
            return output_image

    output = st.empty()

    if selected_model != 'click here':
        random_image, latent = random()

        output_image = manipulate()

        cols = st.beta_columns(len(selected_feature)+1)
        for i, image in enumerate(random_image):
            cols[0].image(image, 'input image ' +
                          str(i+1), use_column_width=True)
        for j in range(len(selected_feature)):
            for image in output_image[selected_feature[j]]:
                cols[j+1].image(image,
                                f'{selected_feature[j]}', use_column_width=True)
                output = image

        with st.sidebar:
            download_button(output, 'Enjoy? Download images here 🎉')

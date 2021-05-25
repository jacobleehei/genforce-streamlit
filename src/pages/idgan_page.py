import streamlit as st
from PIL import Image
import numpy as np
import cv2, os, time, base64, io, torch, sys

if sys.platform == 'darwin':
    sys.path.append('utils')
    import webFunction as web
    import ganFunction.idinvertHandler as idv
else:
    from utils import webFunction as web
    import utils.ganFunction.idinvertHandler as idv



idvTool = idv.indinvert_webObject()

def write():

    model = [
        'styleganinv_ffhq256', 
        'styleganinv_tower256', 
        'styleganinv_bedroom256'
    ]
    
    model_name = st.sidebar.selectbox('Choose your model from here', model)

    idvNode = [
        '📖 Overview',
        '✨ Semantic Diffusion',
        '✨ Interpolation',
        '✨ Manipulation'
    ]

    st.title('In-Domain GAN Inversion')
    st.subheader('by Jiapeng Zhu,  Yujun Shen,  Deli Zhao,  Bolei Zhou ([@GenForce\_](https://genforce.github.io/))')

    st.markdown(
        "[![Paper](https://img.shields.io/badge/Paper-green?logo=xda-developers&logoColor=white)](https://github.com/genforce/interfacegan)  &nbsp;&nbsp;"
        "[![Colab](https://img.shields.io/badge/Colab-F9AB00?logo=google-colab&logoColor=white)](https://colab.research.google.com/github/genforce/idinvert_pytorch/blob/master/docs/Idinvert.ipynb) &nbsp;&nbsp;"
        "[![GitHub](https://img.shields.io/github/stars/genforce/idinvert?style=social&label=Code(TensorFlow)&maxAge=2592000)](https://github.com/genforce/idinvert)&nbsp;&nbsp;"
        "[![GitHub](https://img.shields.io/github/stars/genforce/idinvert_pytorch?style=social&label=Code(PyTorch)&maxAge=2592000)](https://github.com/genforce/idinvert_pytorch)&nbsp;&nbsp;"        
        )
    
    if idvTool.generator[model_name] is None: 
        idvTool.model_name = model_name
        build_idvTool()

    operation = st.selectbox('Please choose the operation you want',list(idvNode), index = 0)
    writePageNode(operation)



def writePageNode(operation):

    if operation == '📖 Overview':
        st.markdown("""

        The GAN inversion task is required not only to reconstruct the target image by pixel values, but also to keep the inverted code in the semantic domain of the original latent space of well-trained GANs. 
        
        For this purpose,In-Domain GAN inversion (IDInvert) is proposed by:
        - First training a novel domain-guided encoder which is able to produce in-domain latent code
        - Then performing domain-regularized optimization which involves the encoder as a regularizer to land the code inside the latent space when being finetuned. 
        
        The in-domain codes produced by IDInvert enable high-quality real image editing with fixed GAN models.

        ## In-Domain GAN semantic diffusion:
        """, unsafe_allow_html=True) 
        with st.spinner('Loading...☕'):
            web.open_gif('img/web/idvpage/diffusion.gif')

            st.subheader('Image editing results:')
            web.open_gif('img/web/idvpage/edit.gif')

            st.subheader('Image editing results:')
            st.video('https://youtu.be/3v6NHrhuyFY')
            
    if operation == '✨ Semantic Diffusion': Semantic_Diffusion()
    if operation == '✨ Manipulation'      : Manipulation()
    if operation == '✨ Interpolation'     : Interpolation()


def Semantic_Diffusion():

    Top, _ = st.beta_columns((1, 0.0001))
    b1, b2 = st.beta_columns((7, 2))
    c1, c2 = st.beta_columns((7, 2))

    result_img = None

    b1.subheader('Step 1: Upload a target image')
    c1.subheader('Step 2: Upload a context image')
    target_img, target_img_cv = uploadImg(b1,'target')
    context_img, context_img_cv = uploadImg(c1,'context')
    


    @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
    def c_align(image):
        path = 'temp_c.png'
        context_img.save(path, 'PNG')
        with st.spinner('Aligning image...⏳'):
            return idv.align_face(path)

    if target_img:
        b2.image(target_img, use_column_width= True)
    if context_img: 
        c2.image(context_img, use_column_width= True)
    
    
    st.subheader('Step 3: Adjust the crop size and diffuse!')
    A1, A2 = st.beta_columns((1,1))
    crop_size = A1.slider('Crop size', 50, 150, 100, 5)
    
    if target_img and context_img:
        revert =  A1.checkbox('revert images')
        if revert: cropImg = cropHelper(target_img, context_img, crop_size, crop_size)
        else: cropImg = cropHelper(context_img, target_img, crop_size, crop_size)

        if A1.button('🚀Comfirm and start generate'):
            if revert:
                with st.spinner('Diffusing the images...⏳'):
                    result_img = idvTool.diffuse(target_img_cv, context_img_cv, crop_size)
            else:
                with st.spinner('Diffusing the images...⏳'):
                    result_img = idvTool.diffuse(context_img_cv, target_img_cv, crop_size)


        if result_img is None: A2.image(cropImg, use_column_width = True)
        else: 
            st.success('diffuse images sucessful🎉')
            A2.image(result_img, use_column_width = True)


def Interpolation():

    a1, a2 = st.beta_columns((5, 2))
    b1, b2 = st.beta_columns((5, 2))

    a1.subheader('Step 1: Upload a target image')
    b1.subheader('Step 2: Upload a context image')
    src_file = a1.file_uploader(label="First image!", type=['jpg', 'png', 'jpeg'])
    tar_file = b1.file_uploader(label="Sencond image!", type=['jpg', 'png', 'jpeg'])

    if src_file:
        src_file = Image.open(src_file)
        src_file.save(os.path.abspath('img/indomain_output/interpolate/context_temp/temp.png'), 'PNG')
        src_file_path = os.path.abspath('img/indomain_output/interpolate/context_temp/temp.png')
        a2.image(src_file, 'First Photo', use_column_width = True)


    if tar_file:
        tar_file = Image.open(tar_file)
        tar_file.save(os.path.abspath('img/indomain_output/interpolate/target_temp/temp.png'), 'PNG')
        tar_file_path = os.path.abspath('img/indomain_output/interpolate/target_temp/temp.png')
        b2.image(tar_file, 'Second Photo', use_column_width = True)

    st.subheader('Step 3: Adjust the step size')
    step = st.slider('step', 10, 50, 20, 10)


    if st.button('🚀Click this button to generate'):
        with st.spinner('Inverting two images...⏳'):
            result = idvTool.interpolation(src_file_path, tar_file_path, step)

        st.success('Interpolate two images sucessful!🎉')
        result_img = []
        result_img_invert = []
        for i, image in enumerate(result):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.imwrite('img/indomain_output/interpolate/result/' + str(i) + '.png',image)
            img = Image.open('img/indomain_output/interpolate/result/' + str(i) + '.png')
            if i == 0: 
                for j in range(100):result_img.append(img)
            result_img.append(img)

        result_img_invert = result_img[::-1]
        for i in range(100): result_img.append(result_img_invert[0])
        for image in result_img_invert: result_img.append(image)
        result_img[0].save('img/indomain_output/interpolate/result/gif.gif', save_all=True, append_images= result_img[1:], optimize=False, duration=5, loop=0)

        file_ = open("img/indomain_output/interpolate/result/gif.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        col1,col2,col3 = st.beta_columns((1,1,1))
        col1.image(src_file,width=200)
        col2.image(tar_file,width=200)
        col3.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="gif" width=200>',
        unsafe_allow_html=True,
        )


def Manipulation(): 

    Top,_ = st.beta_columns((1,0.000001))
    A1, A2 = st.beta_columns((8,3))
    B1, B2 = st.beta_columns((1,0.000001))

    Top.subheader('Step 1: Upload and Invert an Image')
    uploadImg = A1.file_uploader(label="Upload your image!", type=['jpg', 'png', 'jpeg'],accept_multiple_files=False)

    if uploadImg:
        src_file_path = os.path.abspath('img/indomain_output/manipulation/temp/temp.png')
        Image.open(uploadImg).save(src_file_path, 'PNG')
        A2.image(uploadImg, 'uploaded photo', use_column_width= True)

    if A1.button('🚀click here to invert to image'):
        with st.spinner('inverting source image...⏳'):
            idvTool.align_invert(src_file_path)




    @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
    def manipulate():
        return idvTool.manipulation(age, gender, pose, eyeglasses, expression)

    #Show the Output image
    if idvTool.latentCode is not None:
    #Show feature in sidebar    
        with st.sidebar:
            st.subheader('👱🏼 Feature')
            age = st.slider('age', -3.0, 3.0, 0.0, 0.5)
            gender = st.slider('gender', -3.0, 3.0, 0.0, 0.5)
            pose = st.slider('pose', -3.0, 3.0, 0.0, 0.5)
            expression = st.slider('expression', -3.0, 3.0, 0.0, 1.0)
            eyeglasses = st.slider('eyeglasses', -3.0, 3.0, 0.0, 0.5)
        st.subheader('Step 2: Play with the feature in the sidebar')
        old_image, new_image = manipulate()
        col1, col2 = st.beta_columns((1,1))
        col1.image(old_image, 'Input', use_column_width = True)
        col2.image(new_image, 'Output', use_column_width = True)


def uploadImg(column, str):
    
    img_file = column.file_uploader(label=f"Upload the {str} image!", type=['jpg', 'png', 'jpeg'])

    @st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
    def align(image):
        a_path = 'img/indomain_output/temp.png'
        Image.open(image).save(a_path, 'PNG')
        with st.spinner('Aligning image...⏳'):
            return idv.align_face(a_path)

    if img_file:
        aligned_t = Image.fromarray(align(img_file))
        path = f'img/indomain_output/{str}_temp/temp.png'
        aligned_t.save(os.path.abspath(path), 'PNG')
        img_file_cv = cv2.imread(path)

        return aligned_t, img_file_cv

    else: return None, None


def cropHelper(context_img, target_img, crop_width, crop_height):
    img_width, img_height = context_img.size
    crop_img =  context_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
    target_img.paste(crop_img, ((img_width - crop_width)//2,(img_height - crop_height)//2))
    return target_img


def build_idvTool():
    progressbar = st.progress(1)

    with st.spinner('⏳ ...building In-Domain GAN generator'):
        idvTool.build_generator()
        progressbar.progress(50)

    with st.spinner('⏳ ...building In-Domain GAN inverter'):
        idvTool.build_inverter()
        progressbar.progress(100)
    time.sleep(0.5)

    progressbar.empty()
    return idvTool
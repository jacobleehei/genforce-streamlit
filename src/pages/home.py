import streamlit as st
import os, sys

if sys.platform == 'darwin':
        sys.path.append('utils')
        import webFunction, fileDetail
else:
        from utils import webFunction, fileDetail

def write():
        # Download all data files if they aren't already in the working directory.
        for filename in fileDetail.EXTERNAL_DEPENDENCIES.keys():
                webFunction.download_file(filename)

        icon_path = "img/web/icon.gif" 
        webFunction.open_gif(icon_path, 400)

        st.title('Generative Force')
        
        st.markdown(
        "[![GitHub](https://img.shields.io/github/stars/genforce/interfacegan?style=social&label=InterFaceGAN&maxAge=2592000)](https://github.com/genforce/interfacegan)&nbsp;&nbsp;"
        "[![GitHub](https://img.shields.io/github/stars/genforce/idinvert?style=social&label=InDomainGAN(TensorFlow)&maxAge=2592000)](https://github.com/genforce/idinvert)&nbsp;&nbsp;"
        "[![GitHub](https://img.shields.io/github/stars/genforce/higan?style=social&label=HiGAN&maxAge=2592000)](https://github.com/genforce/higan)&nbsp;&nbsp;"
        "[![Follow](https://img.shields.io/twitter/follow/LeeCheungHei1?style=social)](https://www.twitter.com/LeeCheungHei1)"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        Generative Adversarial Networks (GANs) have significantly advanced image synthesis in recent years. 
        
        The rationale behind GANs is to learn the mapping from a latent distribution to the real data through adversarial training.
        
        After learning such a non-linear mapping, GAN is capable of producing photo-realistic images.
        
        ---
        """)

        st.subheader('what this app can do? Try it your self!✨')
        st.selectbox('choose a GAN', ['🎈InterFace GAN', '🎈In-DomainGAN Inversion', '🎈HiGAN'])
        
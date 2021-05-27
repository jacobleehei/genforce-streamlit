[![GitHub](https://img.shields.io/github/stars/genforce/interfacegan?style=social&label=InterFaceGAN&maxAge=2592000)](https://github.com/genforce/interfacegan)&nbsp;&nbsp;
[![GitHub](https://img.shields.io/github/stars/genforce/idinvert?style=social&label=InDomainGAN(TensorFlow)&maxAge=2592000)](https://github.com/genforce/idinvert)&nbsp;&nbsp;
[![GitHub](https://img.shields.io/github/stars/genforce/higan?style=social&label=HiGAN&maxAge=2592000)](https://github.com/genforce/higan)&nbsp;&nbsp;
[![Follow](https://img.shields.io/twitter/follow/LeeCheungHei1?style=social)](https://www.twitter.com/LeeCheungHei1)

# GenForce-Streamlit
This project highlights three powerful Generative model based methods [InterFace GAN](https://genforce.github.io/interfacegan/), [HiGAN](https://genforce.github.io/higan/) and [In-Domain GAN](https://genforce.github.io/idinvert/) for tuning the output image's characteristics, inverting real images and randomly generating faces and hierarchy images. For more information, check out the [Research Initiative on Generative Modeling](https://genforce.github.io/). 

![In-use Animation](https://github.com/jasonleehei/GenForce-Streamlit/blob/main/demo-itf.gif?raw=true "In-use Animation")

If you want to see the full demo of this app, check it out the there:

https://www.youtube.com/watch?v=bbS5gHZubzM&ab_channel=CLOEBE



## How to run this app
The app requires Python 3.6 or 3.7. 
**It is suggested that creating a new virtual environment**, then running:

```
git clone https://github.com/jasonleehei/GenForce-Streamlit.git
cd GenForce-Streamlit
pip install -r requirements.txt
streamlit run app.py
```

## What this app can do?

For InterFaceGAN:
| Function      | 	Supported model
| ------------- |:-------------:|
| Manipulation of real image     | stylegan_ffhq |
| Manipulation of random sampling     | 	stylegan_ffhq, stylegan_celebahq, pggan_celebahq     |

For HiGAN:
| Function      | 	Supported model
| ------------- |:-------------:|
| Manipulation of random sampling     | stylegan_bedroom, stylegan_livingroom, stylegan2_church,
|| stylegan_churchoutdoor, stylegan_tower, stylegan_kitchen,
|| stylegan_bridge |

For In-DomainGAN:
| Function      | 	Supported model
| ------------- |:-------------:|
| Semantic diffusion of real images     | styleganinv_ffhq256, styleganinv_tower256, styleganinv_bedroom256   |
| Interpolation of real images     | 	styleganinv_ffhq256, styleganinv_tower256, styleganinv_bedroom256       |
| Manipulation of real image | styleganinv_ffhq256, styleganinv_tower256, styleganinv_bedroom256   |
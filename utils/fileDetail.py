#This file is for save the files' detail which will be downlaod in the first time

EXTERNAL_DEPENDENCIES = {

    # InterFaceGAN
    "pggan_celebahq.pth" : {
        "path": "gan/interfacegan/InterFaceGAN/models/pretrain/",
        "url": "https://www.dropbox.com/s/t74z87pk3cf8ny7/pggan_celebahq.pth?dl=1",
        "size": 90167
    },
    "stylegan_celebahq.pth": {
        "path": "gan/interfacegan/InterFaceGAN/models/pretrain/",
        "url": "https://www.dropbox.com/s/nmo2g3u0qt7x70m/stylegan_celebahq.pth?dl=1",
        "size": 113456
    },
    "stylegan_ffhq.pth": {
        "path": "gan/interfacegan/InterFaceGAN/models/pretrain/",
        "url": "https://www.dropbox.com/s/qyv37eaobnow7fu/stylegan_ffhq.pth?dl=1",
        "size": 113482
    },

    # In-DomainGAN (face)
    "styleganinv_ffhq256_encoder.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://www.dropbox.com/s/k3nkf73qrue3wix/styleganinv_ffhq256_encoder.pth?dl=1",
        "size": 645955
    },
    "styleganinv_ffhq256_generator.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://www.dropbox.com/s/7zw080o50btlafv/styleganinv_ffhq256_generator.pth?dl=1",
        "size": 116023
    },

    # In-DomainGAN (bedroom)
    "styleganinv_bedroom256_encoder.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://www.dropbox.com/s/k3nkf73qrue3wix/styleganinv_ffhq256_encoder.pth?dl=1",
        "size": 645955
    },
    "styleganinv_bedroom256_generator.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://www.dropbox.com/s/l8axjmsgn03g7g9/styleganinv_bedroom256_generator.pth?dl=1",
        "size": 116023
    },

    # In-DomainGAN (tower)
    "styleganinv_tower256_encoder.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://www.dropbox.com/s/6fkzclzh5kx1p2r/styleganinv_tower256_encoder.pth?dl=1",
        "size": 645955
    },
    "styleganinv_tower256_generator.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://www.dropbox.com/s/9gb8h7xsies1a83/styleganinv_tower256_generator.pth?dl=1",
        "size": 116023
    },

    "vgg16.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://drive.google.com/u/0/uc?id=1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y&export=download",
        "size": 57483
    },

    # higan 
    "stylegan_apartment256_generator.pkl" : {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/j0uob3nhjxk00el/stylegan_apartment256_generator.pth?dl=1",
        "size": 102705
    },
    "stylegan_bedroom256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/h1w7ld4hsvte5zf/stylegan_bedroom256_generator.pth?dl=1",
        "size": 102705
    },
    "stylegan_bridge256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/hj8c91v50efmqac/stylegan_bridge256_generator.pth?dl=1",
        "size": 102705
    },

    "stylegan_churchoutdoor256_generator.pth" : {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/dun5giw5ia2o1ol/stylegan_churchoutdoor256_generator.pth?dl=1",
        "size": 102705
    },
    "stylegan_diningroom256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/kszbnsqtnxxxo9o/stylegan_diningroom256_generator.pth?dl=1",
        "size": 102705
    },
    "stylegan_kitchen256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/rzc48uomq3k0tlg/stylegan_kitchen256_generator.pth?dl=1",
        "size": 102705
    },

    "stylegan_livingroom256_generator.pth" : {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/nzx4kxctirge2wp/stylegan_livingroom256_generator.pth?dl=1",
        "size": 102705
    },
    "stylegan_tower256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/9uubkv0fg8vub9u/stylegan_tower256_generator.pth?dl=1",
        "size": 102705
    },
    "stylegan2_church256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/riop29m4h93cxgi/stylegan2_church256_generator.pth?dl=1",
        "size": 116023
    },
    "order_w_1k.npy": {
        "path": "utils/ganFunction/",
        "url": "https://www.dropbox.com/s/hwjyclj749qtp89/order_w.npy?dl=1",
        "size": 2001
    },

}

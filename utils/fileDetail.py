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
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=7DXi&id=1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO",
        "size": 645955
    },
    "styleganinv_ffhq256_generator.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=snqe&id=1SjWD4slw612z2cXa3-n38JwKZXqDUerG",
        "size": 116023
    },

    # In-DomainGAN (bedroom)
    "styleganinv_bedroom256_encoder.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=6_h8&id=1ebuiaQ7xI99a6ZrHbxzGApEFCu0h0X2s",
        "size": 645955
    },
    "styleganinv_bedroom256_generator.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=ggCf&id=1ka583QwvMOtcFZJcu29ee8ykZdyOCcMS",
        "size": 116023
    },

    # In-DomainGAN (tower)
    "styleganinv_tower256_encoder.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=3vSd&id=1Pzkgdi3xctdsCZa9lcb7dziA_UMIswyS",
        "size": 645955
    },
    "styleganinv_tower256_generator.pth": {
        "path": "gan/idinvert/models/pretrain/",
        "url": "https://drive.google.com/u/0/uc?export=download&confirm=Uju6&id=1lI_OA_aN4-O3mXEPQ1Nv-6tdg_3UWcyN",
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
        "url": "https://ucd7a30c9f97b8cdeab104eef101.dl.dropboxusercontent.com/cd/0/get/BPFvB_s2Lym_gAyalQBepx8CQRjC64-Ulh0JqQnTEqsz49y6HtAwWOzCGZ6gWs1YU7D9GVWHVyuAhWBksw7Em4ecX9D3r5AJcpUIFrmBiOITw_cKM8WZCfQBYFYLY6lRyTcYUJY6roH00v_Dpn3fVFnV/file",
        "size": 102705
    },
    "stylegan_bedroom256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/h1w7ld4hsvte5zf/stylegan_bedroom256_generator.pth?dl=1",
        "size": 102705
    },
    "stylegan_bridge256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://ucaf2ddfb09f108de1ab816e2540.dl.dropboxusercontent.com/cd/0/get/BPFIF0vvLfarD2_eFNpzuDbzqHLc9uSVc8_vXorZ9vShEqIw4MkMgt59PYGtltZjkcfb4kdnW6kRDDpeY5atCx9RQMJ8pXHgy64HRhjFStzMXxN8ckC6SsHyRfbZjiY1GgdWiJLcCTYeB496mHXPIbkx/file",
        "size": 102705
    },

    "stylegan_churchoutdoor256_generator.pth" : {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://uc7c770b55cbb4840f4e1fe27e25.dl.dropboxusercontent.com/cd/0/get/BPExEGOme5mcIVghLp1LC29hfcZZFi1_G-H-ht4cAVIs71D28Rnu5xECyOj03s1C69jJqpCD0sv_zaGGloag8IhIHmyDVmARGmMZf7IST8A_UzNdi-i44pBctTCa-bHtxCMiuZjJbI9V67TGfyImGd3d/file",
        "size": 102705
    },
    "stylegan_diningroom256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://ucd0be48b83bdac86d39bbcf467f.dl.dropboxusercontent.com/cd/0/get/BPHU0lv8Snj4OYY-BeAT7H0hyRlR6VSfXk54G5jn0-5SS0bbMVDpZXqC2RZuuVZybZx_-zZQJamPM6AWYWWWkNNmqg8feIh3hbMmJABsPodgzLOxRrzL1lNTrkwJdAXUv342Vl_U0s-Cs8fMvQUmQnfl/file",
        "size": 102705
    },
    "stylegan_kitchen256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://ucf4f49cea28f2a638cff7297236.dl.dropboxusercontent.com/cd/0/get/BPEbUg89-cMf2sdEToa6M-tFXe9E4POa_wI2R9NDjQmWDwiLcQ2npioGALirlkzb3e1gkRrfLRqxpc056VB0s-S_zsAA4yk5NPKf1_OvGWNgTK1amoRFJYrH_ehm8tlM1mPb3TrGfZBmqBK5dRDbX8GJ/file",
        "size": 102705
    },

    "stylegan_livingroom256_generator.pth" : {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://uc5e2e6377a3cbdd49a719392172.dl.dropboxusercontent.com/cd/0/get/BPHtHA_VKvusiMAQyhg5GBWxMXIw9x675dUyLoQdj0ixzokn-1hWHlQMKR7VqT_sDC3_Xi4gdemVlWBVlVPXFoef7ADOMvkSO2TBHChdCkn71T-0yhztIF_MVo93Icbo7svM1yGay7r3uYOHe5zWSGAN/file",
        "size": 102705
    },
    "stylegan_tower256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://uc0e9a28c4d2a43666116c056ef3.dl.dropboxusercontent.com/cd/0/get/BPGD4ehEkF0qxwyz4kh0L-7c4fAqO8u-yV4sABX9fIUk3VJWGJiKmMsrtsqY0ewadk0hgd4ZAbupm-XESVUguOKIYu5U-TBGMayNYkCbZ2Ge4R9l9dxu7_KtrmDGyTT2Sz26Fzx_BO5FfvxUAdlbQmeC/file",
        "size": 102705
    },
    "stylegan2_church256_generator.pth": {
        "path": "gan/higan/models/pretrain/pytorch/",
        "url": "https://www.dropbox.com/s/l8i9qpgv82zvi7g/stylegan2_church256_generator.pth?dl=0?dl=1",
        "size": 116023
    },
    "order_w_1k.npy": {
        "path": "utils/ganFunction/",
        "url": "https://www.dropbox.com/s/hwjyclj749qtp89/order_w.npy?dl=1",
        "size": 2001
    },

}

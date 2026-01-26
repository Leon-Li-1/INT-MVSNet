InceptionMamba-MVS: Plug-and-Play Modules for Unsupervised Multi-View Stereo

Details are described in our paper:
> Improving Unsupervised Multi-View Stereo via Distinctive Feature Matching and Multi-Granular Cost Aggregation
>
> Liangliang Li, Guihua Liu, Feng Xu, Wenjin Liao

This repository implements two novel plug-and-play modules that significantly boost unsupervised multi-view stereo (MVS) performance without requiring ground-truth depth supervision:
 
 ✅ **InceptionMamba**  
  Robust multi-granular feature matching via state-space modeling

 ✅ **InceptionConv3D**  
  Hierarchical cost aggregation across depth planes

*If there are any errors in our code, please feel free to ask your questions.*

## ⚙ Setup
#### 1. Recommended environment
- PyTorch 2.4
- Python 3.12

#### 2. DTU Dataset

**Training Data**. Download [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). 
Unzip them and put the `Depth_raw` to `dtu_training` folder. The structure is just like:
```
dtu_training                          
       ├── Cameras
       ├── Depths
       ├── Depths_raw
       └── Rectified
```
**Testing Data**. Download [DTU testing data](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) and unzip it. The structure is just like:
```
origin                          
	├── scan1                
	├── scan2   
       		├── cams
			├── 00000000_cam.txt
			├── 00000001_cam.txt
			├── ...
		├── images
			├── 00000000.jpg
			├── 00000001.jpg
			├── ...
		└── pair.txt
	├── ...
```


## 📊 Testing

#### DTU testing

**Note:** `pretrained_model/model.ckpt` is the model trained on DTU without any finetuning.

```
bash ./scripts/dtu_test.sh
```

## ⏳ Training

#### DTU training

```
bash ./scripts/dtu_train.sh
```


## ⚖ Citation
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{xiong2023cl,
  title={Improving Unsupervised Multi-View Stereo via Distinctive Feature Matching and Multi-Granular Cost Aggregation},
  author={Liangliang Li, Guihua Liu, Feng Xu, Wenjin Liao},
  booktitle={  },
  pages={  },
  year={2026}
}
```

## 👩‍ Acknowledgements

Thanks to [CL-MVSNet](https://KaiqiangXiong.github.io/CL-MVSNet/), [MVSNet](https://github.com/YoYo000/MVSNet), [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch), [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet), [RC-MVSNet](https://github.com/Boese0601/RC-MVSNet),[UniMVSNet](https://github.com/prstrive/UniMVSNet), and [InceptionNeXt]( https://github.com/sail-sg/inceptionnext/),

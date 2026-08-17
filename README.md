Enhancing Matching Energy Distinctiveness and Multi-Granular Cost Aggregation for Unsupervised Multi-View Stereo

Details are described in our paper:
> From the perspective of multi-view feature matching, we propose two Inception-style plug-and-play modules. Our approach comprehensively examines features at different granularities to enhance the discriminability of the cost volume across various depth hypothesis planes, thereby improving model performance. By integrating these two modules into CL-MVSNet, we propose INT-MVSNet.
>We greatly appreciate the foundational work of CL-MVSNet, upon which we have made only minor improvements.
> 
> Liangliang Li, Guihua Liu, Hongwei Quan, Xiaoying Hong

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

## 👩‍ Acknowledgements

Thanks to [CL-MVSNet](https://KaiqiangXiong.github.io/CL-MVSNet/), [MVSNet](https://github.com/YoYo000/MVSNet), [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch), [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet), [RC-MVSNet](https://github.com/Boese0601/RC-MVSNet),[UniMVSNet](https://github.com/prstrive/UniMVSNet), and [InceptionNeXt]( https://github.com/sail-sg/inceptionnext/),

# FORECAST-CLSTM: A New Convolutional LSTM Network for Cloudage Nowcasting


by [Chao Tan](https://), Xin Feng, Jianwu Long, and Li Geng.       

This repository contains the source code and dataset for FORECAST-CLSTM, provided by [Chao Tan](https://).

The paper is avaliable for download [here](https://arxiv.org/ftp/arxiv/papers/1905/1905.07700.pdf).
Click [here](https://) for more details.

***

## Dataset

Our SCMD2016 dataset is available for download at [TianYiCloud(2.5GB)](https://cloud.189.cn/t/aqqy2uYZviMj) or [BaiduCloud(2.5GB)](https://pan.baidu.com/s/1s2QkY_p9mKltoB0rsf9oLA) (extraction code: ssby).           
SCMD dataset is a brand new cloudage nowcasting dataset for deep learning research.
It contains 20000 grayscale image sequences for training and another 3500 image sequences for testing.
You can get the SCMD2016 dataset at any time but only for scientific research. 
At the same time, please cite our work when you use the SCMD dataset.

The mnist dataset in npz format can be download [here](https://s3.amazonaws.com/img-datasets/mnist.npz).


        
## Prerequisites
* Python 3.5
* PyTorch >= 0.4.0
* opencv 0.4
* PyQt 4
* numpy

  
## Train
1. For moving-mnist training, please download mnist.npz dataset and place it in  ```./data``` folder, for cloudage nowcasting training, please download and unzip SCMD2016 dataset and place it in ```./data``` folder.
2. For moving-mnist dataset, run ```python trainer_mnist.py --model "FORECAST_CLSTM_M" --epochs 100  --train-batch 16 --gpu-ids "0" --checkpoint "checkpoint/forecast_clstm_m"``` to start training.
3. For scmd2016 dataset, run ```python trainer_scmd.py --model "FORECAST_CLSTM_S" --epochs 100  --train-batch 16 --gpu-ids "0" --checkpoint "checkpoint/forecast_clstm_s""``` to start training.

## Citation

@inproceedings{      
&nbsp;&nbsp;&nbsp;&nbsp;  title={{FORECAST-CLSTM}: A New Convolutional LSTM network for Cloudage Nowcasting},         
&nbsp;&nbsp;&nbsp;&nbsp;  author={Tan, Chao and Feng, Xin and Long, Jianwu and Geng, Li},         
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle={VCIP},        
&nbsp;&nbsp;&nbsp;&nbsp;  year={2018},        
&nbsp;&nbsp;&nbsp;&nbsp;  note={to appear},       
}

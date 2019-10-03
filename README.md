# FORECAST-CLSTM: A New Convolutional LSTM Network for Cloudage Nowcasting


by Chao Tan, Xin Feng, Jianwu Long, and Li Geng


***

## Dataset

Our SCMD2016 dataset is available for download at [Amazon Drive(2.0GB)]().
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

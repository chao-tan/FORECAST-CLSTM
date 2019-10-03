# -*- coding: utf-8 -*-
import torch
import dataset
import cv2

model_clstm_m = "checkpoint/clstm_m/model_best.pth"
model_clstm_s = "checkpoint/clstm_s/model_best.pth"
mdoel_forecast_clstm_m = "checkpoint/forecast_clstm_m/model_best.pth"
mdoel_forecast_clstm_s = "checkpoint/forecast_clstm_s/model_best.pth"
model_forecast_clstm_forecaster="checkpoint/forecast_clstm_forecaster/model_best.pth"



def demo_mnist(model_path):
    model = torch.load(model_path)
    mnist = dataset.MovingMnist_Generation(digtnum=2,
                                           width=64,
                                           height=64,
                                           seq_length=9)
    x_batch,y_batch = mnist.next_batch(batch_size=1,
                                       next_seqlen=1,
                                       return_one=False,
                                       norm=False)
    x_batch = torch.from_numpy(x_batch).float()

    output = model.forward(x_batch)
    output = output.detach().cpu().numpy()
    cv2.imwrite("demo_mnist.png",output[0][0][0])



def demo_scmd(model_path):
    model = torch.load(model_path)
    scmd = dataset.SCDMD_Generation()
    x_batch,y_batch = scmd.next_batch(batchsize=1)
    x_batch = torch.from_numpy(x_batch).float()

    output = model.forward(x_batch)
    output = output.detach().cpu().numpy()
    cv2.imwrite("demo_scmd.png",output[0][0][0])
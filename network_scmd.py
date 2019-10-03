# -*- coding: utf-8 -*-
import torch
from torch.nn import functional as F
from torch import nn
import tools


class CLSTM_S(nn.Module):
    def __init__(self):
        super(CLSTM_S,self).__init__()
        self.steps = 5

        self.spconv1 = tools.FSCONV2D(in_channels=1,
                                      out_channels=16,
                                      kernel_size=(3,3),
                                      padding=(1,1),
                                      stride=(1,1))
        self.convlstm1 = tools.ConvLSTM(input_size=(200,200),input_dim=16,hidden_dim=16,kernel_size=(3,3))
        self.spool1 = tools.FSPOOL2D()

        self.spconv2 = tools.FSCONV2D(in_channels=16,
                                      out_channels=32,
                                      kernel_size=(3,3),
                                      padding=(1,1),
                                      stride=(1,1))
        self.convlstm2 = tools.ConvLSTM(input_size=(100,100),input_dim=32,hidden_dim=32,kernel_size=(3,3))
        self.spool2 = tools.FSPOOL2D()

        self.spconv3 = tools.FSCONV2D(in_channels=32,
                                      out_channels=64,
                                      kernel_size=(3,3),
                                      padding=(1,1),
                                      stride=(1,1))
        self.convlstm3 = tools.ConvLSTM(input_size=(50,50),input_dim=64,hidden_dim=64,kernel_size=(3,3))

        #############################################################
        self.dconv1 = tools.FSDCONV2D(in_channels=64,
                                      out_channels=32,
                                      kernel_size=(3,3),
                                      padding=(1,1),
                                      stride=(1,1))
        self.upool1 = tools.FSUNPOOLING()

        self.dconv2 = tools.FSDCONV2D(in_channels=32,
                                      out_channels=16,
                                      kernel_size=(3,3),
                                      padding=(1,1),
                                      stride=(1,1))
        self.upool2 = tools.FSUNPOOLING()

        self.dconv3 = tools.FSDCONV2D(in_channels=16,
                                      out_channels=16,
                                      kernel_size=(3,3),
                                      padding=(1,1),
                                      stride=(1,1))

        self.outlayer = tools.FSDCONV2D(in_channels=16,
                                        out_channels=1,
                                        kernel_size=(1,1),
                                        padding=(0,0),
                                        stride=(1,1))

        self.batchnorm1 = nn.BatchNorm3d(num_features=self.steps)
        self.batchnorm1_c = nn.BatchNorm3d(num_features=self.steps)
        self.batchnorm2 = nn.BatchNorm3d(num_features=self.steps)
        self.batchnorm2_c = nn.BatchNorm3d(num_features=self.steps)
        self.batchnorm3 = nn.BatchNorm3d(num_features=self.steps)
        self.batchnorm3_c = nn.BatchNorm3d(num_features=self.steps)

        self.batchnorm_d1 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d2 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d3 = nn.BatchNorm3d(num_features=1)


    def forward(self, x):
        x = self.batchnorm1(self.spconv1(x))
        x = F.leaky_relu(x,inplace=True)
        x,_ = self.convlstm1(x)
        x = F.leaky_relu(self.batchnorm1_c(x),inplace=True)
        x,ind1 = self.spool1(x)

        x = self.batchnorm2(self.spconv2(x))
        x = F.leaky_relu(x,inplace=True)
        x,_ = self.convlstm2(x)
        x = F.leaky_relu(self.batchnorm2_c(x),inplace=True)
        x,ind2 = self.spool2(x)

        x = self.batchnorm3(self.spconv3(x))
        x = F.leaky_relu(x,inplace=True)
        x,_ = self.convlstm3(x)
        x = F.leaky_relu(self.batchnorm3_c(x),inplace=True)

        ##############################################################

        x = x[:,4:5,:,:,:]
        x = self.batchnorm_d1(self.dconv1(x))
        x = F.leaky_relu(x,inplace=True)
        x = self.upool1(x,ind2[-1])

        x = self.batchnorm_d2(self.dconv2(x))
        x = F.leaky_relu(x,inplace=True)
        x = self.upool2(x,ind1[-1])

        x = self.batchnorm_d3(self.dconv3(x))
        x = F.leaky_relu(x,inplace=True)

        x = self.outlayer(x)

        return x






class FORECAST_CLSTM_S(nn.Module):
    def __init__(self):
        super(FORECAST_CLSTM_S,self).__init__()

        self.spconv1_1 = tools.FSCONV2D(in_channels=1,
                                        out_channels=16,
                                        kernel_size=(3,3),
                                        padding=(1,1),
                                        stride=(1, 1))
        self.spconv1_2 = tools.FSDCONV2D(in_channels=16,
                                         out_channels=16,
                                         kernel_size=(3,3),
                                         padding=(1,1),
                                         stride=(1,1))
        self.convlstm1 = tools.ConvLSTM(input_size=(200,200), input_dim=16, hidden_dim=16, kernel_size=(3,3))
        self.spool1 = tools.FSPOOL2D()

        self.spconv2_1 = tools.FSDCONV2D(in_channels=16,
                                         out_channels=32,
                                         kernel_size=(3,3),
                                         padding=(1,1),
                                         stride=(1,1))
        self.spconv2_2 = tools.FSDCONV2D(in_channels=32,
                                         out_channels=32,
                                         kernel_size=(3,3),
                                         padding=(1,1),
                                         stride=(1,1))
        self.convlstm2 = tools.ConvLSTM(input_size=(100,100),input_dim=32,hidden_dim=32,kernel_size=(3,3))
        self.spool2 = tools.FSPOOL2D()

        self.spconv3_1 = tools.FSDCONV2D(in_channels=32,
                                         out_channels=64,
                                         kernel_size=(3,3),
                                         padding=(1,1),
                                         stride=(1,1))
        self.spconv3_2 = tools.FSDCONV2D(in_channels=64,
                                         out_channels=64,
                                         kernel_size=(3,3),
                                         padding=(1,1),
                                         stride=(1,1))
        self.convlstm3 = tools.ConvLSTM(input_size=(50,50),input_dim=64,hidden_dim=64,kernel_size=(3,3))

        # batchnorm for encode network
        self.batchnorm1_1 = nn.BatchNorm3d(num_features=5)
        self.batchnorm1_2 = nn.BatchNorm3d(num_features=5)
        self.batchnorm1_c = nn.BatchNorm3d(num_features=5)

        self.batchnorm2_1 = nn.BatchNorm3d(num_features=4)
        self.batchnorm2_2 = nn.BatchNorm3d(num_features=4)
        self.batchnorm2_c = nn.BatchNorm3d(num_features=4)

        self.batchnorm3_1 = nn.BatchNorm3d(num_features=3)
        self.batchnorm3_2 = nn.BatchNorm3d(num_features=3)
        self.batchnorm3_c = nn.BatchNorm3d(num_features=3)

        #################################################################################################
        # decoder1
        self.decoder1_1_1 = tools.FSDCONV2D(in_channels=16,
                                            out_channels=16,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder1_1_2 = tools.FSDCONV2D(in_channels=16,
                                            out_channels=16,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.outlayer1 = tools.FSDCONV2D(in_channels=16,
                                         out_channels=1,
                                         kernel_size=(1,1),
                                         padding=(0,0),
                                         stride=(1,1))

        # batchnorm for decoder1
        self.batchnorm_d1_1_1 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d1_1_2 = nn.BatchNorm3d(num_features=1)

        ################################################################################################
        # decoder2
        self.decoder2_1_1 = tools.FSDCONV2D(in_channels=32,
                                            out_channels=32,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder2_1_2 = tools.FSDCONV2D(in_channels=32,
                                            out_channels=16,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder2_upool1 = tools.FSUNPOOLING()
        self.decoder2_2_1 = tools.FSDCONV2D(in_channels=16,
                                            out_channels=16,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder2_2_2 = tools.FSDCONV2D(in_channels=16,
                                            out_channels=16,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.outlayer2 = tools.FSDCONV2D(in_channels=16,
                                        out_channels=1,
                                        kernel_size=(1,1),
                                        padding=(0,0),
                                        stride=(1,1))

        # batchnorm for decoder2
        self.batchnorm_d2_1_1 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d2_1_2 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d2_2_1 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d2_2_2 = nn.BatchNorm3d(num_features=1)

        ###################################################################################################
        # decoder3
        self.decoder3_1_1 = tools.FSDCONV2D(in_channels=64,
                                            out_channels=64,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder3_1_2 = tools.FSDCONV2D(in_channels=64,
                                            out_channels=32,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder3_unpool1 = tools.FSUNPOOLING()
        self.decoder3_2_1 = tools.FSDCONV2D(in_channels=32,
                                            out_channels=32,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder3_2_2 = tools.FSDCONV2D(in_channels=32,
                                            out_channels=16,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder3_unpool2 = tools.FSUNPOOLING()
        self.decoder3_3_1 = tools.FSDCONV2D(in_channels=16,
                                            out_channels=16,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.decoder3_3_2 = tools.FSDCONV2D(in_channels=16,
                                            out_channels=16,
                                            kernel_size=(3,3),
                                            padding=(1,1),
                                            stride=(1,1))
        self.outlayer3 = tools.FSDCONV2D(in_channels=16,
                                        out_channels=1,
                                        kernel_size=(1,1),
                                        padding=(0,0),
                                        stride=(1,1))

        # batchnorm for decoder3
        self.batchnorm_d3_1_1 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d3_1_2 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d3_2_1 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d3_2_2 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d3_3_1 = nn.BatchNorm3d(num_features=1)
        self.batchnorm_d3_3_2 = nn.BatchNorm3d(num_features=1)


        #############################################################################################
        # integration module
        self.inte1 = tools.FSCONV2D(in_channels=3,
                                    out_channels=16,
                                    kernel_size=(1,1),
                                    padding=(0,0),
                                    stride=(1,1))
        self.out_final = tools.FSCONV2D(in_channels=16,
                                        out_channels=1,
                                        kernel_size=(1,1),
                                        padding=(0,0),
                                        stride=(1,1))



    def forward(self, x):
        x = self.batchnorm1_1(self.spconv1_1(x))
        x = F.leaky_relu(x,inplace=True)
        x = self.batchnorm1_2(self.spconv1_2(x))
        x = F.leaky_relu(x,inplace=True)
        x,_ = self.convlstm1(x)
        x = F.leaky_relu(self.batchnorm1_c(x))
        x_encoder1 = x[:,4:5,:,:,:]

        x,ind1 = self.spool1(x[:,1:5,:,:,:])
        x = self.batchnorm2_1(self.spconv2_1(x))
        x = F.leaky_relu(x,inplace=True)
        x = self.batchnorm2_2(self.spconv2_2(x))
        x = F.leaky_relu(x,inplace=True)
        x,_ = self.convlstm2(x)
        x = F.leaky_relu(self.batchnorm2_c(x))
        x_encoder2 = x[:,3:4,:,:,:]

        x,ind2 = self.spool2(x[:,1:4,:,:,:])
        x = self.batchnorm3_1(self.spconv3_1(x))
        x = F.leaky_relu(x,inplace=True)
        x = self.batchnorm3_2(self.spconv3_2(x))
        x = F.leaky_relu(x,inplace=True)
        x,_ = self.convlstm3(x)
        x = F.leaky_relu(self.batchnorm3_c(x))
        x_encoder3 = x[:,2:3,:,:,:]

        ################################################################################
        # decoder1
        x_decoder1 = self.batchnorm_d1_1_1(self.decoder1_1_1(x_encoder1))
        x_decoder1 = F.leaky_relu(x_decoder1)
        x_decoder1 = self.batchnorm_d1_1_2(self.decoder1_1_2(x_decoder1))
        x_decoder1 = F.leaky_relu(x_decoder1)
        x_decoder1 = self.outlayer1(x_decoder1)

        #################################################################################
        # decoder2
        x_decoder2 = self.batchnorm_d2_1_1(self.decoder2_1_1(x_encoder2))
        x_decoder2 = F.leaky_relu(x_decoder2)
        x_decoder2 = self.batchnorm_d2_1_2(self.decoder2_1_2(x_decoder2))
        x_decoder2 = F.leaky_relu(x_decoder2)
        x_decoder2 = self.decoder2_upool1(x_decoder2,ind1[-1])
        x_decoder2 = self.batchnorm_d2_2_1(self.decoder2_2_1(x_decoder2))
        x_decoder2 = F.leaky_relu(x_decoder2)
        x_decoder2 = self.batchnorm_d2_2_2(self.decoder2_2_2(x_decoder2))
        x_decoder2 = F.leaky_relu(x_decoder2)
        x_decoder2 = self.outlayer2(x_decoder2)

        ####################################################################################
        # decoder3
        x_decoder3 = self.batchnorm_d3_1_1(self.decoder3_1_1(x_encoder3))
        x_decoder3 = F.leaky_relu(x_decoder3)
        x_decoder3 = self.batchnorm_d3_1_2(self.decoder3_1_2(x_decoder3))
        x_decoder3 = F.leaky_relu(x_decoder3)
        x_decoder3 = self.decoder3_unpool1(x_decoder3,ind2[-1])
        x_decoder3 = self.batchnorm_d3_2_1(self.decoder3_2_1(x_decoder3))
        x_decoder3 = F.leaky_relu(x_decoder3)
        x_decoder3 = self.batchnorm_d3_2_2(self.decoder3_2_2(x_decoder3))
        x_decoder3 = F.leaky_relu(x_decoder3)
        x_decoder3 = self.decoder3_unpool2(x_decoder3,ind1[-1])
        x_decoder3 = self.batchnorm_d3_3_1(self.decoder3_3_1(x_decoder3))
        x_decoder3 = F.leaky_relu(x_decoder3)
        x_decoder3 = self.batchnorm_d3_3_2(self.decoder3_3_2(x_decoder3))
        x_decoder3 = F.leaky_relu(x_decoder3)
        x_decoder3 = self.outlayer3(x_decoder3)

        x_out = torch.cat((x_decoder1,x_decoder2,x_decoder3),dim=2)
        x_out = self.inte1(x_out)
        x_out = self.out_final(x_out)

        return x_out


class MLP_S(nn.Module):
    def __init__(self):
        super(MLP_S,self).__init__()

        self.mlp1 = nn.Linear(in_features=5,out_features=256,bias=True)
        self.mlp2 = nn.Linear(in_features=256,out_features=256,bias=True)
        self.outlayer = nn.Linear(in_features=256,out_features=1,bias=True)

        self.batchnorm_mlp1 = nn.BatchNorm1d(num_features=256)
        self.batchnorm_mlp2 = nn.BatchNorm1d(num_features=256)

    # input x with size(batchsize,timestep,channels,1)
    # output with size (batchsize,1,channels,1)
    def forward(self, x,indx=100,indy=100):
        x = x.squeeze(-1)

        x = self.batchnorm_mlp1(self.mlp1(x))
        x = F.leaky_relu(x,inplace=True)
        x = F.dropout(x,p=0.5)

        x = self.batchnorm_mlp2(self.mlp2(x))
        x = F.leaky_relu(x,inplace=True)
        x = F.dropout(x,p=0.5)

        x = self.outlayer(x)
        x = x.unsqueeze(1).unsqueeze(1)
        return x



class FC_LSTM_S(nn.Module):
    def __init__(self):
        super(FC_LSTM_S,self).__init__()

        self.lstm1 = nn.LSTM(input_size=1,
                             hidden_size=256,
                             num_layers= 1)

        self.lstm2 = nn.LSTM(input_size=256,
                             hidden_size=256,
                             num_layers=1)

        self.lstm3 = nn.LSTM(input_size=256,
                             hidden_size=1,
                             num_layers=1)




    def forward(self, x,indx,indy):
        x,_ = self.lstm1(x)
        x = F.dropout(F.leaky_relu(x),0.5)
        x,_ = self.lstm2(x)
        x = F.dropout(F.leaky_relu(x),0.5)
        x,_ = self.lstm3(x)
        x = x[:,-1,:].unsqueeze(1).unsqueeze(1)

        # return (batchsize,1,1,1)
        return x


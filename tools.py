# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size,input_dim,hidden_dim,kernel_size,bias):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0]//2, kernel_size[1]//2)
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())
        else:
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width))



class ConvLSTM(nn.Module):

    def __init__(self, input_size,              # exp:(200,200)
                 input_dim,                     # exp:1
                 hidden_dim,                    # exp:32
                 kernel_size,                   # exp:(3,3)
                 return_one=False,
                 bias=True):

        super(ConvLSTM, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.use_gpu = torch.cuda.is_available()
        self.return_one = return_one
        self.bias = bias

        self.cell = ConvLSTMCell(input_size=(self.height,self.width),
                                 input_dim=self.input_dim,
                                 hidden_dim=self.hidden_dim,
                                 kernel_size=self.kernel_size,
                                 bias=self.bias)


    # input_tensor with shape (batchsize,steps,channels,height,width)
    # input hidden_state is [h,c] list
    # for h and c with shape (batchsize,out_channels,height,width)
    # if return_one = True, return (batchsize,channels,height,width)
    # if return_one = False, return (batchsize,steps,channels,height,width)
    def forward(self, input_tensor,
                hidden_state=None):

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        seq_len = input_tensor.size(1)

        h, c = hidden_state
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell(input_tensor=input_tensor[:, t, :, :, :],
                             cur_state=[h, c])
            output_inner.append(h)

        layer_output = torch.stack(output_inner, dim=1)

        if self.return_one:
            return layer_output[:,-1,:,:,:],[h,c]
        else:
            return layer_output, [h,c]


    def _init_hidden(self, batch_size):
        return self.cell.init_hidden(batch_size)






class FSCONV2D(nn.Module):
    def __init__(self,in_channels,                                  #Input channels of the samples
                 out_channels,                                      #Output channels of the samples
                 kernel_size,                                       #Kernel size of the convolution operation
                 stride,                                            #Stride of the convolution operation
                 padding,                                           #Padding of the convolution operation
                 bias=True):                                             #If use bias for every kernel

        super(FSCONV2D,self).__init__()
        self.input_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x):                                           # input x with shape:(batchsize,steps,channels,width,height)

        x_split = torch.split(x,1,dim=1)
        out =[]
        for i in range(len(x_split)):
            out.append(self.conv(x_split[i].squeeze(dim=1)))

        # output with shape:(batchsize,steps,channels,width,height)
        return torch.stack(out,dim=1)







class FSDCONV2D(nn.Module):
    def __init__(self,in_channels,                                  #Input channels of the samples
                 out_channels,                                      #Output channels of the samples
                 kernel_size,                                       #Kernel size of the convolution operation
                 stride,                                            #Stride of the convolution operation
                 padding,                                           #Padding of the convolution operation
                 bias=True):                                             #If use bias for every kernel

        super(FSDCONV2D,self).__init__()
        self.input_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv = nn.ConvTranspose2d(in_channels=self.input_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x):                                           # input x with shape:(batchsize,steps,channels,width,height)

        x_split = torch.split(x,1,dim=1)
        out =[]
        for i in range(len(x_split)):
            out.append(self.conv(x_split[i].squeeze(dim=1)))

        # output with shape:(batchsize,steps,channels,width,height)
        return torch.stack(out,dim=1)






"The Implement of First Seprate Pooling Network"
class FSPOOL2D(nn.Module):
    def __init__(self, kernel_size=(2,2),
                 stride=(2,2)):
        super(FSPOOL2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.pooling = nn.MaxPool2d(kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    return_indices=True)

    def forward(self, x):                                           # input x with shape:(batchsize,steps,channels,width,height)
        x_split = torch.split(x, 1, dim=1)
        out = []
        ind = []
        for i in range(len(x_split)):
            c, indx = self.pooling(x_split[i].squeeze(dim=1))
            out.append(c)
            ind.append(indx)

        # output with shape:(batchsize,steps,channels,width,height)
        return torch.stack(out, dim=1), ind



class FSUNPOOLING(nn.Module):
    def __init__(self,kernel_size=(2,2)):
        super(FSUNPOOLING,self).__init__()
        self.kernel_size = kernel_size

        self.unpooling = nn.MaxUnpool2d(kernel_size=self.kernel_size)

    def forward(self, x,ind):
        x_split = torch.split(x,1,dim=1)
        out=[]
        for i in range(len(x_split)):
            out.append(self.unpooling(x_split[i].squeeze(1),ind))

        return torch.stack(out,dim=1)


class FORECASTER_LOSS(nn.Module):
    def __init__(self):
        super(FORECASTER_LOSS,self).__init__()

    def forward(self, output,ground):
        output = output.view(-1)
        ground = ground.view(-1)
        gap = torch.abs(output-ground)
        weight = (output+ground-gap)/2
        weight = 1-weight/255.0
        weight = torch.exp(weight)
        loss = torch.mean(weight*(output-ground)*(output-ground))
        return loss

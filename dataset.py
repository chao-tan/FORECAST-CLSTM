# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import cv2


def load_mnist(traing_num=50000):
    dat = np.load("data/mnist.npz")
    X = dat['x_train'][:traing_num]
    Y = dat['y_train'][:traing_num]
    X_test = dat['x_test']
    Y_test = dat['y_test']
    Y = Y.reshape((Y.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    return X, Y, X_test, Y_test


def move_step(v0, p0, bounding_box):
    xmin, xmax, ymin, ymax = bounding_box
    assert (p0[0]>=xmin) and (p0[0]<=xmax) and (p0[1]>=ymin) and (p0[1]<=ymax)
    v = v0.copy()
    assert v[0] != 0.0 and v[1] != 0.0
    p = v0 + p0
    while (p[0]<xmin) or (p[0]>xmax) or (p[1]<ymin) or (p[1]>ymax):
        vx, vy = v
        x, y = p
        dist = np.zeros((4,))
        dist[0] = abs(x-xmin) if ymin <= (xmin-x)*vy/vx+y<=ymax else np.inf
        dist[1] = abs(x-xmax) if ymin <= (xmax-x)*vy/vx+y<=ymax else np.inf
        dist[2] = abs((y-ymin)*vx/vy) if xmin <= (ymin-y)*vx/vy+x<=xmax else np.inf
        dist[3] = abs((y-ymax)*vx/vy) if xmin <= (ymax-y)*vx/vy+x<=xmax else np.inf
        n = np.argmin(dist)
        if n == 0:
            v[0] = -v[0]
            p[0] = 2*xmin-p[0]
        elif n == 1:
            v[0] = -v[0]
            p[0] = 2*xmax-p[0]
        elif n == 2:
            v[1] = -v[1]
            p[1] = 2*ymin-p[1]
        elif n == 3:
            v[1] = -v[1]
            p[1] = 2*ymax-p[1]
        else:
            assert False
    return v, p



class MovingMNISTIterator(object):
    def __init__(self):
        self.mnist_train_img, self.mnist_train_label,self.mnist_test_img, self.mnist_test_label = load_mnist()

    def sample(self, digitnum,
               width,
               height,
               seqlen,
               batch_size,
               index_range=(0, 50000)):
        """"""
        """

        :param digitnum: The num of the digits
        :param width: The width of the generated images
        :param height: The height of the generated images
        :param seqlen: The length of the image sequence
        :param index_range: by default
        :return:
        """
        character_indices = np.random.randint(low=index_range[0], high=index_range[1],size=(batch_size, digitnum))
        angles = np.random.random((batch_size, digitnum)) * (2 * np.pi)
        magnitudes = np.random.random((batch_size, digitnum)) * (5 - 3) + 3
        velocities = np.zeros((batch_size, digitnum, 2), dtype='float32')
        velocities[..., 0] = magnitudes * np.cos(angles)
        velocities[..., 1] = magnitudes * np.sin(angles)
        xmin = 14.0
        xmax = float(width) - 14.0
        ymin = 14.0
        ymax = float(height) - 14.0
        positions = np.random.uniform(low=xmin, high=xmax,size=(batch_size, digitnum, 2))
        seq = np.zeros((seqlen, batch_size, 1, height, width), dtype='uint8')
        for i in range(batch_size):
            for j in range(digitnum):
                ind = character_indices[i, j]
                v = velocities[i, j, :]
                p = positions[i, j, :]
                img = self.mnist_train_img[ind].reshape((28, 28))
                for k in range(seqlen):
                    topleft_y = int(p[0] - img.shape[0] / 2)
                    topleft_x = int(p[1] - img.shape[1] / 2)
                    seq[k, i, 0, topleft_y:topleft_y + 28, topleft_x:topleft_x + 28] = np.maximum(seq[k, i, 0, topleft_y:topleft_y + 28, topleft_x:topleft_x + 28],img)
                    v, p = move_step(v, p, [xmin, xmax, ymin, ymax])
        return seq



class MovingMnist_Generation(object):
    def __init__(self,digtnum, width, height, seq_length):
        self.digtnum = digtnum
        self.width = width
        self.height = height
        self.seq_length = seq_length

    def next_batch(self,batch_size,next_seqlen=1,return_one=True,norm=False):
        movingmnist = MovingMNISTIterator()


        sample =  movingmnist.sample(digitnum=self.digtnum,
                                     width=self.width,
                                     height=self.height,
                                     seqlen=self.seq_length+next_seqlen,
                                     batch_size=batch_size)
        sample = np.transpose(sample,(1,0,2,3,4))


        x_batch = sample[:,0:self.seq_length,:,:,:]
        y_batch = sample[:,self.seq_length:(self.seq_length+next_seqlen),:,:,:]

        if return_one is True and next_seqlen == 1:
            y_batch = np.reshape(y_batch,(batch_size,1,self.width,self.height))

        # return the x_batch with shape(batchsize,seq_length,channels,width,height)
        # return the y_batch with shape(batchsize,seq_length,channels,width,height) or (batchsize,channels,width,height) when y_batch has only one timestep
        if norm:
            return x_batch/255.0 , y_batch/255.0
        else:
            return x_batch,y_batch



class SCMD_Generation(object):
    def __init__(self,seq_length=5,next_seq=1,isTrain=True,return_one=True,norm=False,baseline=False):
        self.seq_length = seq_length    # the length of the squence for training
        self.next_seq = next_seq
        self.is_train = isTrain
        self.return_one = return_one
        self.norm = norm
        self.baseline = baseline
        self.train_root = "data/SCMD2016/TRAIN"
        self.test_root = "data/SCMD2016/TEST"
        self.data_length = 0


    def next_batch(self,batchsize):
        x_batch = np.ndarray(shape=(batchsize,self.seq_length,1,200,200),dtype=np.float32)
        y_batch = np.ndarray(shape=(batchsize,1,1,200,200),dtype=np.float32)

        if self.is_train:
            datalist = os.listdir(self.train_root)
        else:
            datalist = os.listdir(self.test_root)
        self.data_length = len(datalist)

        random_order = random.sample(range(1,self.data_length),batchsize)
        if self.is_train:
            root_path = self.train_root
        else:
            root_path = self.test_root

        for i in range(batchsize):
            for k in range(self.seq_length):
                x_batch[i,k,0,:,:] = cv2.imread(root_path+"/"+str(random_order[i])+"/"+str(k+1)+".png",cv2.IMREAD_GRAYSCALE)
            y_batch[i,0,0,:,:] = cv2.imread(root_path+"/"+str(random_order[i])+"/"+str(self.seq_length+1)+".png",cv2.IMREAD_GRAYSCALE)

        x_batch,y_batch = x_batch,y_batch

        if self.baseline:
            x_batch = x_batch[:,:,:,100,100]
            y_batch = y_batch[:,:,:,100,100]

        if self.norm:
            x_batch = x_batch/255.0
            y_batch = y_batch/255.0


        return x_batch,y_batch


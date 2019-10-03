# encoding: utf-8
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from network_mnist import FORECAST_CLSTM_M,CLSTM_M
from utils import Bar, Logger, AverageMeter
import dataset


parser = argparse.ArgumentParser(description='Cloudage Nowcasting Training')
parser.add_argument('--model', type=str, default="FORECAST_CLSTM_M",help='traing model, optional in FORECAST_CLSTM_M and CLSTM_M')
parser.add_argument('--epochs', type=int, default=100,metavar='N',help='number of total epochs to run')
parser.add_argument('--train-batch', default=16,type=int, metavar='N',help='train batchsize')
parser.add_argument('--test-batch', default=1,type=int, metavar='N',help='test batchsize')
parser.add_argument('--train-iters',default="2000",type=int,help="training iterations")
parser.add_argument('--test-iters',default="200",type=int,help="test iterations")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, default=[70],nargs='+',help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu-ids', default="0", type=str, help='traing gpu ids')
parser.add_argument('--checkpoint', default="checkpoint/forecast_clstm_m", type=str)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(manualSeed)

if args.model == "FORECAST_CLSTM_M":
    net = FORECAST_CLSTM_M
elif args.model == 'CLSTM_M':
    net = CLSTM_M
else:
    pass

if not os.path.exists(os.path.join(args.checkpoint)):
    os.makedirs(os.path.join(args.checkpoint))



best_loss = 10000000000

mmnist = dataset.MovingMnist_Generation(digtnum=2,
                                        width=64,
                                        height=64,
                                        seq_length=9)


def main():
    global best_loss

    print("==> creating model ...")
    model = net()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    title = "mmnist"

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss'])

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss  = train(model, criterion, optimizer)
        test_loss = test( model, criterion)

        logger.append([state['lr'], train_loss, test_loss])

        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        if is_best:
            torch.save(model,os.path.join(args.checkpoint,"model_best.pth"))


    logger.close()
    logger.plot()

    print('Best Loss:')
    print(best_loss)




def train(model, criterion, optimizer):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=int(args.train_iters/args.train_batch))
    for i in range(int(args.test_iters/args.train_batch)):
        data_time.update(time.time() - end)

        inputs,targets = mmnist.next_batch(batch_size=args.train_batch,
                                            next_seqlen=1,
                                            return_one=False,
                                            norm=False)
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))

        file = open(os.path.join(args.checkpoint,"tlogs.txt"),"a+")
        file.write(str(loss.item()))
        file.write("\n")
        file.close()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=i + 1,
                    size=int(args.train_iters/args.train_batch),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg


def test(model, criterion):
    global best_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=int(args.test_iters/args.test_batch))
    for i in range(int(args.test_iters/args.test_batch)):
        data_time.update(time.time() - end)

        inputs, targets = mmnist.next_batch(batch_size=args.train_batch,
                                            next_seqlen=1,
                                            return_one=False,
                                            norm=False)
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=i + 1,
                    size=int(args.test_iters/args.test_batch),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()
    bar.finish()

    return losses.avg

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()



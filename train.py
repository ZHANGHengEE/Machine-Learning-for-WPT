import argparse
import os
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import time  # 引入time模块
from torch.utils.data import DataLoader

from unet import Generator, Discriminator
from dataset import train_test_split, MagnetDataset


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count if self._count != 0 else 0

    def getval(self):
        return self._avg

    def __str__(self):
        if not hasattr(self, 'val'):
            return 'None.'
        return str(self.getval())


def train(epoch, args, loader, model, criterion, optimizerG, optimizerD):
    netG, netD = model
    netG.train()
    netD.train()
    for i, (_, data_txt, label, label_txt) in enumerate(loader):
        data_txt, label, label_txt =\
        data_txt.to(args.device), label.to(args.device), label_txt.to(args.device)


        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        # netD.zero_grad()
        # Format batch
        # real_cpu = data[0].to(device)
        b_size = data_txt.size(0)
        # gt = torch.full((b_size,), 1., dtype=torch.float, device=args.device)
        # Forward pass real batch through D
        label = label.unsqueeze(1)
        # output = netD(label, label_txt, data_txt).view(-1)
        # # Calculate loss on all-real batch
        # errD_real = criterion(output, gt)
        # # Calculate gradients for D in backward pass
        # errD_real.backward()
        # D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # Generate fake image batch with G
        # Classify all fake batch with D
        # output = netD(fake_img.detach(), fake_txt.detach(), data_txt).view(-1)
        # # Calculate D's loss on the all-fake batch
        # errD_fake = criterion(output, gt)
        # # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        # errD_fake.backward()
        # D_G_z1 = output.mean().item()
        # # Compute error of D as sum over the fake and the real batches
        # errD = errD_real + errD_fake
        # # Update D
        # optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # gt.fill_(1.)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output_img, output_txt = netG(data_txt)
        # Calculate G's loss based on this output
        err1 = criterion(output_img, label)
        err2 = criterion(output_txt, label_txt)
        errG = err1 + err2
        # Calculate gradients for G
        errG.backward()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_G: %.4f'
                  % (epoch, args.epochs, i, len(loader),
                     errG.item()))


@torch.no_grad()
def evaluate(epoch, args, loader, model, metric_fn):
    netG, netD = model
    netG.eval()
    netD.eval()
    metric_img = AverageMeter()
    metric_txt = AverageMeter()
    for itr, (_, data_txt, label, label_txt) in enumerate(loader):
        data_txt, label, label_txt =\
        data_txt.to(args.device), label.to(args.device), label_txt.to(args.device)
        predict_img, predict_txt = netG(data_txt)
        loss_img = metric_fn(predict_img.flatten(1), label.flatten(1))
        # loss_txt = torch.abs(predict_txt - label_txt).mean()
        loss_txt = (torch.abs(predict_txt - label_txt) / torch.abs(label_txt)).mean()
        
        metric_img.update(loss_img.item())
        metric_txt.update(loss_txt.item())


    print(f'Average evaluation Error => IMG: {metric_img.getval()}, TXT: {metric_txt.getval()}')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  )  
    print()  
    
    with open('./weights/Error.txt', 'a') as ff:     # 打开test.txt   如果文件不存在，创建该文件。
        # ff.write("var")  # 把变量var写入test.txt。这里var必须是str格式，如果不是，则可以转一下。
        ff.write('%d,'%(epoch) )
        ff.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        ff.write(", ")
        ff.write('%s%f%s%f\r\n'%("IMG,",metric_img.getval(),",TXT,",metric_txt.getval()))  #X,Y,Z为整型变量，则写入后内容为firstX_Y_Zhours :(变量分别用值代替)  

        
    return metric_img.getval()


@torch.no_grad()
def save_results(args, loader, model):
    print('Saving...')
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    model.eval()
    outputs = []
    targets = []
    for itr, (_, data_txt, label, label_txt) in enumerate(loader):
        if itr >= 100: break
        data_txt = data_txt.to(args.device)
        # data = data.unsqueeze(1)
        predict_img, predict_txt = model(data_txt)
        #predict = predict_img.cpu().numpy()
        predict = predict_img.squeeze().cpu().numpy()
        outputs.append(predict)
        targets.append(label)
    outputs = np.concatenate(outputs, 0)
    targets = np.concatenate(targets, 0)
    for idx, (data, label) in enumerate(zip(outputs, targets)):
        # data = data.transpose(1, 2, 0)
        # label = label.transpose(1, 2, 0)
        data =   data * 255.0
        label =   label * 255.0
        cv2.imwrite(os.path.join(args.output_path, f'{idx}_predict.png'), data)
        cv2.imwrite(os.path.join(args.output_path, f'{idx}_label.png'), label)   

    

def main(args):
    torch.autograd.set_detect_anomaly(True) 
    model_g = Generator().to(args.device)
    model_d = Discriminator().to(args.device)
    loss_fn = nn.BCELoss()
    metric_fn = nn.L1Loss()  # nn.MSELoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    optimizerG = optim.Adam(model_g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(model_d.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    trainset, testset = train_test_split(args.data_path)
    trainset = MagnetDataset(trainset, imgsz=args.imgsz)
    testset = MagnetDataset(testset, imgsz=args.imgsz)
    print(f'Data length: {len(trainset)}, {len(testset)}')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    best_metric = 0.415
    for i in range(args.epochs):
        train(i, args, train_loader, (model_g, model_d), loss_fn, optimizerG, optimizerD)
        metric = evaluate(i, args, test_loader, (model_g, model_d), metric_fn)
        # scheduler.step(metric)

        if metric < best_metric:
            best_metric = metric
            ckpt = {
                '(model_g, model_d)': (model_g.state_dict(),model_d.state_dict()),
                # 'optimizer': optimizer.state_dict()
                '(optimizerG, optimizerD)': (optimizerG.state_dict(),optimizerD.state_dict())
                # 'optimizerD': optimizerD.state_dict()
            }
            if not os.path.exists(args.weight_path):
                os.mkdir(args.weight_path)
            torch.save(ckpt, os.path.join(args.weight_path, 'best.pt'))

    save_results(args, test_loader, model_g)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data-path', default='./data/dataCapNew1D230226_30k', type=str)
    parser.add_argument('--output-path', default='./data/outputs', type=str)
    parser.add_argument('--weight-path', default='./weights', type=str)
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs')
    parser.add_argument('--imgsz', type=tuple, default=(512, 512), help='Image size')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', '-l', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of channels')
    parser.add_argument('--gpus', type=str, default='0')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # args.device =  torch.device('cpu')
    print(args.device)

    main(args)

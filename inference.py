import argparse
import time
import os
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from dataset import train_test_split, MagnetDataset
from unet import Generator, Discriminator

@torch.no_grad()
def infer(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists('./data/outputs_10'):
        os.mkdir('./data/outputs_10')
    # model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear).to(args.device)
    # model.eval()
    # weights = torch.load(args.weight_path, map_location=args.device)
    # model.load_state_dict(weights['model'])

    model_g = Generator().to(args.device)
    model_g.eval()
    weights = torch.load(args.weight_path, map_location=args.device)
    model_g.load_state_dict(weights['(model_g, model_d)'][0])

    # model_g = Generator().to(args.device)
    # model_g.eval()
    # model_d = Discriminator().to(args.device)
    # model_d.eval()
    # model= (model_g,model_d)
    # weights = torch.load(args.weight_path, map_location=args.device)
    # model.load_state_dict(weights['(model_g, model_d)'])

    datalist = os.listdir(args.data_path)
    # datalist = list(filter(lambda x: x.endswith('Ori.png') and int(x.split('_')[2]) > 5, datalist))
    datalist = list(filter(lambda x: x.endswith('Ori.png'), datalist))   
    datalist = list(map(lambda x: '_'.join(x.split('_')[:-1]), datalist))
    datalist = sorted(datalist, key=lambda x: int(x.split('_')[0][-5:]))
    testset = MagnetDataset(datalist, imgsz=args.imgsz)
    loader = DataLoader(testset, batch_size=1)
    print('Start inference...')

    for idx, (_, data_txt, label, label_txt) in enumerate(loader):
        start = time.time()
        prefix = '_'.join(str( round(float(data_txt[0][i].item()),2) )  for i in range(7))
        # Itx=round(float(data_txt[0][5].item()),1)

        # data, data_txt_gpu = data.to(args.device), data_txt.to(args.device)
        # data = data.unsqueeze(1)
        # data_txt_gpu = data_txt_gpu[None] #  / 400.
        # predict_img, predict_txt = model(data, data_txt_gpu)
        data_txt_gpu = data_txt.to(args.device)
        data_txt_gpu = data_txt_gpu[None] #  / 400.        
        predict_img, predict_txt = model_g(data_txt_gpu)

        predict_txt = predict_txt.squeeze().cpu().numpy()
        M, L1, Q1, R1 = predict_txt[0:4]

        # M = M * (6.3442e-05 - 2.9278e-08) + 2.9278e-08    
        # L1 = L1 * (4.6380e-02 - 6.5287e-07) + 6.5287e-07
        # Q1 = Q1 * (3.6427e+02 - 4.9272e+00) + 4.9272e+00
         ######random 10000
        # M = M * (4.520e-05 -0) + 0
        # L1 = L1 * (1.11088297e-01- 0) + 0
        # Q1 = Q1 * (3.25e+02-0) + 0
         ######random 30000
###### dataCapNew1D230226 30k random 
            # (float(string[3]) - 0) / (3.59e-05 -0),            
            # (float(string[4]) -0) / (1.24e-01 -0),
            # (float(string[6]) -0) / (3.46e+02 - 0),
            # (float(string[7]) - 0) / (3.06e+03- 0),
            # (float(string[12]) - 0) / (2.60e-02 - 0),
            # (float(string[14]) - 0) / (2.64e-09 - 0)
            ## 3: M,   4: L1,  6:Q1,   7:R1,   12:Maxsurface(T),   14:Minsurface(T) 

        M = M * (3.59e-05 -0) + 0
        L1 = L1 * (1.24e-01- 0) + 0
        Q1 = Q1 * (3.46e+02-0) + 0
        R1 = R1 * (3.06e+03-0) + 0
        # L2 = 142.216 * 1e-6
        # Q2 = 19.236
        L2 = 2.2945419621407068E-4
        Q2 = 15.691502054675711

        eff = cal_efficiency(M, L1, L2, Q1, Q2)
        end = time.time()
        total_time = end - start
        prefix += '_' + str(round(eff, 3)) + '_' + str(round(total_time, 4))
        prefix +=  '_' + str(round(M*1000000, 3))+ '_' + str(round(L1*1000, 3))+ '_' + str(round(Q1, 2)) +'_' + str(round(R1*1000, 3))

        predict = predict_img.squeeze().cpu().numpy()
        line_mask = draw_line(predict, predict_txt[-1], predict_txt[-2])
        predict = (predict / predict.max() * 255).astype(np.uint8)
        # label = label.numpy()
        # label = (label / label.max() * 255).astype(np.uint8)
        predict = cv2.applyColorMap(predict, cv2.COLORMAP_JET)
        predict[line_mask == 255] = [255, 255, 255]
        predict = cv2.cvtColor(predict, cv2.COLOR_RGB2BGR)
        # label = cv2.applyColorMap(label, cv2.COLORMAP_JET)
        if data_txt[0][0] == 60 and data_txt[0][1] == 0.95 and data_txt[0][2] == 0.1 and data_txt[0][4] == 0.2:
            cv2.imwrite(os.path.join(args.output_path, f'{prefix}_predict.png'), predict)
            # cv2.imwrite(os.path.join(args.output_path, f'{prefix}_label.png'), label)
        elif data_txt[0][0] == 60 and data_txt[0][1] == 1 and data_txt[0][2] == 0.1 and data_txt[0][4] == 1:
            cv2.imwrite(os.path.join('./data/outputs_10', f'{prefix}_predict.png'), predict)


def cal_efficiency(M, L1, L2, Q1, Q2):
    k2 = M ** 2 / (L1 * L2)
    return k2 * Q1 * Q2 / (1 + k2 * Q1 * Q2)


def draw_line(img_gray, min_val, max_val):
    img_gray = 255 - (img_gray * 255).astype(np.uint8)
    # img_gray = (((img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())) * 255).astype(np.uint8)
    mask = np.ones_like(img_gray) * 255
    #unet
    # min_val = min_val * (4.75949139e-09 - 1.22639357e-12) + 1.22639357e-12
    # max_val = max_val * (1.33987507e-02 - 1.49683161e-05) + 1.49683161e-05
    #10000 random
    # min_val = min_val * (4.87e-09 - 0) + 0
    # max_val = max_val * (2.72085891e-02 - 0) + 0 
    # #10k random
    # min_val = min_val * (2.64e-09 - 0) + 0
    # max_val = max_val * (2.60e-02 - 0) + 0 
    #30k random
    min_val = min_val * (2.64e-09 - 0) + 0
    max_val = max_val * (2.60e-02 - 0) + 0 


    ratio = int((27 * 1e-6 - min_val) / (max_val - min_val) * 255)
    mask[img_gray == ratio] = 0
    
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 13))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 5))
    msk = cv2.medianBlur(mask, 3)
    msk = cv2.erode(msk, kernel1)
    msk = cv2.erode(msk, kernel2)
    # Skeletonization-like operation in OpenCV
    msk = cv2.ximgproc.thinning(~msk)
    msk[0, :] = 0
    msk[:, 0] = 0
    msk[len(msk) - 1, :] = 0
    msk[:, len(msk) - 1] = 0
    
    return msk


def test_draw():
    data_name = './data/dataPaper230222ori/LLsim00188_60_1.000_0.100_64_0.200_150_218_Ori.png'
    label_name = './data/dataPaper230222ori/LLsim00188_60_1.000_0.100_64_0.200_150_218_ResBW.png'
    img = cv2.imread(data_name, 0)[50:-50, 50:-50]
    label = cv2.imread(label_name, 0)[50:-50, 50:-50]
    label = img - label
    txt_name = './data/dataPaper230222ori/LLsim00188_60_1.000_0.100_64_0.200_150_218_TabSave.txt'
    with open(txt_name, 'r') as f:
        string = f.readline().strip().split(',')
    max_val = (float(string[12]) - 1.49683161e-05) / (1.33987507e-02 - 1.49683161e-05)
    min_val = (float(string[14]) - 1.22639357e-12) / (4.75949139e-09 - 1.22639357e-12)
    line_mask = draw_line(label / 255., min_val, max_val)

    predict = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    predict[line_mask == 255] = [255, 255, 255]
    cv2.imwrite('tmp.png', predict)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='./data/dataPaper230222ori', type=str)
    parser.add_argument('--output-path', default='./data/outputs_02', type=str)
    parser.add_argument('--weight-path', default='./weights/bestCapNew30kRand.pt', type=str)
    parser.add_argument('--imgsz', type=tuple, default=(512, 512), help='Image size')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of channels')
    parser.add_argument('--gpus', type=str, default='0')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  )
    infer(args)
    # test_draw()

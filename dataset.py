import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import cv2


class MagnetDataset(Dataset):

    def __init__(self, data_list, imgsz=(256, 256)):
        self.data_list = data_list
        self.suffix_data = '_Ori.png'
        self.suffix_label = '_ResBW.png'
        self.suffix_txt = '_TabSave.txt'
        self.imgsz = imgsz

    def __getitem__(self, idx):
        data_name = os.path.join('./data/dataPaper230222ori', self.data_list[idx] + self.suffix_data)
        label_name = os.path.join('./data/dataPaper230222ori', self.data_list[idx] + self.suffix_label)
        txt_name = os.path.join('./data/dataPaper230222ori', self.data_list[idx] + self.suffix_txt)

        basic_info = self.data_list[idx].split('_')[1:8]
        with open(txt_name, 'r') as f:
            string = f.readline().strip().split(',')
        
        assert len(string) > 3, print(string, txt_name)
        basic_info = np.array(list(map(lambda x: float(x), basic_info)))
        # print(basic_info)
        target = [
            #####2700
            # (float(string[3]) - 2.9278e-08) / (6.3442e-05 - 2.9278e-08),
            # (float(string[4]) - 6.5287e-07) / (4.6380e-02 - 6.5287e-07),
            # (float(string[6]) - 4.9272e+00) / (3.6427e+02 - 4.9272e+00),
            # (float(string[7]) - 3.7756e-03) / (1.2194e+03 - 3.7756e-03),
            # (float(string[12]) - 1.49683161e-05) / (1.33987507e-02 - 1.49683161e-05),
            # (float(string[14]) - 1.22639357e-12) / (4.75949139e-09 - 1.22639357e-12)
            #######1000
            # (float(string[3]) - 2.93207068e-08) / (4.53314766e-05 - 2.93207068e-08),
            # (float(string[4]) - 8.31796457e-07) / (1.17574091e-01 - 8.31796457e-07),
            # (float(string[6]) - 4.28578645e+00) / (3.36197635e+02 - 4.28578645e+00),
            # (float(string[7]) - 1.06819881e-02) / (2.83104730e+03 - 1.06819881e-02),
            # (float(string[12]) - 1.21391291e-05) / (2.55935244e-02 - 1.21391291e-05),
            # (float(string[14]) - 1.44151321e-13) / (4.87230392e-09 - 1.44151321e-13)


            # (float(string[3]) - 2.99184641e-08) / (4.23488059e-05 - 2.99184641e-08),
            # (float(string[4]) - 8.32384864e-07) / (1.07384579e-01 - 8.32384864e-07),
            # (float(string[6]) - 9.11907800e+00) / (2.97934135e+02 - 9.11907800e+00),
            # (float(string[7]) - 1.07068864e-02) / (2.27491627e+03- 1.07068864e-02),
            # (float(string[12]) - 1.80122260e-05) / (2.08905616e-02 - 1.80122260e-05),
            # (float(string[14]) - 3.50622115e-13) / (4.75872665e-09 - 3.50622115e-13)

            ###### 10000 random 
            # (float(string[3]) - 0) / (4.520e-05 -0),
            # (float(string[4]) -0) / (1.11088297e-01 -0),
            # (float(string[6]) -0) / (3.25e+02 - 0),
            # (float(string[7]) - 0) / (2.75813944e+03- 0),
            # (float(string[12]) - 0) / (2.72085891e-02 - 0),
            # (float(string[14]) - 0) / (4.87e-09 - 0)

             ###### 20000 random 
            # (float(string[3]) - 0) / (4.6e-05 -0),            
            # (float(string[4]) -0) / (1.2e-01 -0),
            # (float(string[6]) -0) / (3.3e+02 - 0),
            # (float(string[7]) - 0) / (2.8e+03- 0),
            # (float(string[12]) - 0) / (2.8e-02 - 0),
            # (float(string[14]) - 0) / (4.9e-09 - 0)

            #   ###### 30000 random 
            # (float(string[3]) - 0) / (4.6e-05 -0),            
            # (float(string[4]) -0) / (1.2e-01 -0),
            # (float(string[6]) -0) / (3.5e+02 - 0),
            # (float(string[7]) - 0) / (2.8e+03- 0),
            # (float(string[12]) - 0) / (2.8e-02 - 0),
            # (float(string[14]) - 0) / (4.9e-09 - 0)
            # ### 3: M,   4: L1,  6:Q1,   7:R1,   12:Maxsurface(T),   14:Minsurface(T) 

            #   ###### dataCapNew1D230226 10000 random 
            # (float(string[3]) - 0) / (3.59e-05 -0),            
            # (float(string[4]) -0) / (1.24e-01 -0),
            # (float(string[6]) -0) / (3.28e+02 - 0),
            # (float(string[7]) - 0) / (3.06e+03- 0),
            # (float(string[12]) - 0) / (2.55e-02 - 0),
            # (float(string[14]) - 0) / (2.64e-09 - 0)
            # ## 3: M,   4: L1,  6:Q1,   7:R1,   12:Maxsurface(T),   14:Minsurface(T) 

              ###### dataCapNew1D230226 30k random 
            (float(string[3]) - 0) / (3.59e-05 -0),            
            (float(string[4]) -0) / (1.24e-01 -0),
            (float(string[6]) -0) / (3.46e+02 - 0),
            (float(string[7]) - 0) / (3.06e+03- 0),
            (float(string[12]) - 0) / (2.60e-02 - 0),
            (float(string[14]) - 0) / (2.64e-09 - 0)
            ## 3: M,   4: L1,  6:Q1,   7:R1,   12:Maxsurface(T),   14:Minsurface(T) 



        ]
        img = cv2.imread(data_name, 0)[50:-50, 50:-50]
        label = cv2.imread(label_name, 0)[50:-50, 50:-50]
        #label = img - label
        # cv2.imwrite('tmp.png', label)
        img = cv2.resize(img, self.imgsz) / 255.0
        label = cv2.resize(label, self.imgsz) / 255.0
        # img = img.transpose(2, 0, 1) / 255.0
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRA)
        # label = label.transpose(2, 0, 1) / 255.0
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(basic_info, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.data_list)


def train_test_split(dataset_path, ratio=0.7):
    datalist = os.listdir(dataset_path)
    datalist = list(filter(lambda x: x.endswith('Ori.png'), datalist))
    print(len(datalist))
    data_names = list(map(lambda x: '_'.join(x.split('_')[:-1]), datalist))
    train_len = int(len(data_names) * ratio)
    trainset, testset = random_split(dataset=data_names, lengths=[train_len, len(data_names) - train_len])
    return list(trainset), list(testset)


if __name__ == '__main__':
    trainset, testset = train_test_split('./data/dataPaper230222ori')
    targets = []
    for item in trainset:
        d_path = os.path.join('./data/dataPaper230222ori', item + '_TabSave.txt')
        basic_info = item.split('_')[1:5]
        with open(d_path, 'r') as f:
            string = f.readline().strip().split(',')
        basic_info = list(map(lambda x: float(x), basic_info))
        target = [float(string[x]) for x in [3, 4, 6, 7, 12, 14]]
        targets.append(target)
    targets = np.array(targets)
    print(targets.shape)
    print(np.max(targets, 0))
    print(np.min(targets, 0))
    # dataset = MagnetDataset(trainset)
    # data, data2, label, label2 = next(iter(dataset))
    # print(data2, label2)
    # assert False
    # cv2.imwrite('tmp_raw.png', label)
    # # data: 512 * 512 * 3
    # # color_map: 256 * 3
    # gray_values = np.arange(256, dtype=np.uint8)
    # # color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
    # # color_to_gray_map = dict(zip(color_values, gray_values))
    # color_map = cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).squeeze()
    # label = gray_values[np.argmin((np.abs(label[:, :, None, :] - color_map[None, None, :, :])).sum(-1), -1)]
    # # label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
    # # label = cv2.LUT(label, color_map)
    # cv2.imwrite('tmp.png', label)
    # img = cv2.applyColorMap(label, cv2.COLORMAP_JET)
    # cv2.imwrite('tmp_rec.png', img)
    # # print(color_map)

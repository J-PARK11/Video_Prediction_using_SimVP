import os
import argparse
import numpy as np
import cv2
from skimage.transform import resize

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
from utils import print_log

'''
230317 dataset3 : python make_caltech.py --strides=5 --save_name='dataset3' --num_set=4 --num_seq=4 --freq=20 --in_frame=10 --out_frame=10
230317 dataset4 : python make_caltech.py --strides=5 --save_name='dataset4' --num_set=6 --num_seq=10 --freq=20 --in_frame=10 --out_frame=10
230318 dataset5 : python make_caltech.py --strides=5 --save_name='dataset5' --num_set=6 --num_seq=15 --freq=20 --in_frame=10 --out_frame=10

'''


def create_parser():
    parser = argparse.ArgumentParser(description='Make Caltech Kernel.')
    # Set-up parameters
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='caltech', choices=['mmnist', 'taxibj','kth','caltech'])
    parser.add_argument('--freq', default=20, type=int, help='Num of total Frames')
    parser.add_argument('--strides', default=3, type=int, help='Strides of video')
    parser.add_argument('--current', default=0, type=int, help='Start point of video')
    parser.add_argument('--in_frame', default=10, type=int, help='Num of input frame')
    parser.add_argument('--out_frame', default=10, type=int, help='Num of output frame')
    parser.add_argument('--num_set', default=2, type=int)
    parser.add_argument('--num_seq', default=2, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_name', default='dataset1')

    # model parameters
    parser.add_argument('--in_shape', default=[10, 3, 128 , 160], type=int,nargs='*')

    return parser

# caltech 비디오 프레임 데이터 만들기
class mk_caltech:
    def __init__(self, args):
        super(mk_caltech, self).__init__()
        self.args = args
        self.root = self.args.data_root
        self.path = os.path.join(self.root, 'caltech_raw/')
        self.num_workers = self.args.num_workers
        self.current = self.args.current
        self.freq = self.args.freq
        self.strides = self.args.strides
        self.in_frame = self.args.in_frame
        self.out_frame = self.args.out_frame
        self.save_name = self.args.save_name
        self.in_shape = self.args.in_shape
        self.num_set = self.args.num_set
        self.num_seq = self.args.num_seq
    
    # Video Frames set 만들기.
    def make_video_set(self, type):
        train_root = os.path.join(self.path, type)
        x_set, y_set = [], []
        if type == 'Train':
            num_set = self.num_set
            num_seq = self.num_seq
        else:
            num_set = 4
            num_seq = 4

        set_list = list(os.listdir(train_root))
        print(f'{type} - len set_list : {len(set_list)}')
        # tqdm Progessing bar to set list
        for set_epoch in tqdm(set_list[:num_set]):    
            print(f'{set_epoch} :')
            set_path = os.path.join(train_root, set_epoch)
            seq_list = list(os.listdir(set_path))
            for seq_file in seq_list[:num_seq]:   
                seq_path = os.path.join(set_path, seq_file)
                
                # capture video to frame
                capture = self.capture_video(seq_path)
                # xy_split
                x, y = self.xy_split(capture)
                print('x :',x.shape,'y :',y.shape)
                x_set.append(x)
                y_set.append(y)
                
        # Print Video Config
        config_video = cv2.VideoCapture(seq_path)
        self.print_vidcap_info(config_video)
        
        # save data
        x_set = np.concatenate(x_set)
        y_set = np.concatenate(y_set)
        print(f'x_set.shape : {x_set.shape}, y_set.shape : {y_set.shape}')

        np.savez(self.root + 'caltech/' + self.save_name + type + '.npz',
                 x=x_set, y=y_set)

    # 한 seq 데이터에 대해 video frame 나누기 : (1841, 128, 160, 3)
    def capture_video(self, path):
        vidcap = cv2.VideoCapture(path)
        if self.current > 0 :   
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, self.current)
        
        frames=[]
        while True:
            for j in range(self.strides):
                success, image = vidcap.read()
            if not success:
                break
            image = self.process_im(image, (128, 160)) / 255.0
            frames.append(image)
        frames = np.array(frames)

        return frames   

    def xy_split(self, capture):
        bundle_size = capture.shape[0] // (self.in_frame + self.out_frame)
        x, y = [], []
        for iter in range(bundle_size): # 167
            start = iter*self.freq
            x.append(capture[start : start+self.in_frame])
            y.append(capture[start+self.in_frame : start+self.in_frame+self.out_frame])
        x = np.array(x)
        y = np.array(y)

        return x, y
    
    # process image
    def process_im(self, im, desired_sz):
        target_ds = float(desired_sz[0])/im.shape[0]
        im = resize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))), preserve_range=True)
        d = int((im.shape[1] - desired_sz[1]) / 2)
        im = im[:, d:d+desired_sz[1]]
        return im

    # 비디오 Config 정보 출력.
    def print_vidcap_info(self,vid_cap):
        config_path = self.root + 'caltech/' + self.save_name + '.txt'
        f = open(config_path, 'w')
        f.write(f'height: {vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n')
        f.write(f'width: {vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}\n')
        f.close()
    

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = mk_caltech(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.make_video_set('Train')
    exp.make_video_set('Test')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
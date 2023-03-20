import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import TensorDataset
from utils import print_log

'''
230314 dataset1 : python make_kth.py --freq=20 --strides=5 --current=0 --out_frame=10
230315 dataset2 : python make_kth.py --freq=20 --strides=1 --current=0 --out_frame=10 --save_name='dataset_s1'
'''

def create_parser():
    parser = argparse.ArgumentParser(description='Make kth Kernel.')
    # Set-up parameters
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='kth')
    parser.add_argument('--freq', default=20, type=int, help='Num of total Frames')
    parser.add_argument('--strides', default=5, type=int, help='Strides of video')
    parser.add_argument('--current', default=0, type=int, help='Start point of video')
    parser.add_argument('--out_frame', default=10, type=int, help='Num of output frame')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--save_name', default='dataset')

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 120, 160], type=int,nargs='*')

    return parser

class mk_kth:
    def __init__(self, args):
        super(mk_kth, self).__init__()
        self.args = args
        self.root = self.args.data_root
        self.num_workers = self.args.num_workers
        self.freq = self.args.freq
        self.strides = self.args.strides
        self.current = self.args.current
        self.out_frame = self.args.out_frame
        self.save_name = self.args.save_name

    def make_kth_dataset(self):
        
        path = os.path.join(self.root, 'kth_raw/')
        class_list = os.listdir(path)

        # Video Capture to frame
        data = list()
        for fname in class_list:
            fpath = path + fname
            flist = list(os.listdir(fpath))
            for video in flist:
                img_path = fpath + '/' + video
                capture = self.capture_video(img_path)
                data.append(capture)
        
        # Print Video Config
        config_video = cv2.VideoCapture(img_path)
        self.print_vidcap_info(config_video)
        
        # Normalization & shuffle dataset
        dataset = np.array(data) / 255.0
        np.random.shuffle(dataset)

        # [B, T, H, W, C] --> [B, T, C, H, W]
        dataset = np.swapaxes(dataset,2,4)
        dataset = np.swapaxes(dataset,3,4)

        num_of_out = 10 + self.out_frame
        train_x, train_y = dataset[:500,:10], dataset[:500,10:num_of_out]
        test_x, test_y = dataset[500:,:10], dataset[500:,10:num_of_out]
        
        np.savez(self.root + 'kth/' + self.save_name + '.npz',
                 train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    def capture_video(self, path):
        vidcap = cv2.VideoCapture(path)
        if self.current > 0 :
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, self.current)
        
        frames=[]
        for i in range(self.freq):
            for j in range(self.strides):
                success, image = vidcap.read()
            image = image[:,:,0]
            frames.append(image[:,:,np.newaxis])
        frames = np.array(frames)
        return frames

    # 비디오 Config 정보 출력.
    def print_vidcap_info(self,vid_cap):
        config_path = self.root + 'kth/' + self.save_name + '.txt'
        f = open(config_path, 'w')
        fps = vid_cap.get(cv2.CAP_PROP_FPS) / self.strides
        f.write(f'초당 프레임 수: {fps}\n')
        f.write(f'height: {vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n')
        f.write(f'width: {vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)}\n')
        f.write(f'총 프레임 수: {vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)}\n')
        f.write(f'현재 프레임 번호: {vid_cap.get(cv2.CAP_PROP_POS_FRAMES)}\n')
        f.write(f'노출: {vid_cap.get(cv2.CAP_PROP_EXPOSURE)}\n')
        length_of_video = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) / vid_cap.get(cv2.CAP_PROP_FPS)
        f.write(f'영상 길이: {length_of_video}s\n')
        stride_of_frame = 1 / fps
        f.write(f'프레임 당 시간 간격: {stride_of_frame}s\n')
        f.close()

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = mk_kth(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.make_kth_dataset()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
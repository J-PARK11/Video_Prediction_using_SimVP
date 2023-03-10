import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from API import *
from model import SimVP
import vis_tool



'''
230309 Try1 : python visualization.py --epochs=200 --res_dir='./results/230309_OG_SimVP' --fig_dir='./figure/230309_OG_SimVP' --batch_size=16 --val_batch_size=16 --dataname='mmnist'


'''
def create_parser():
    parser = argparse.ArgumentParser(description='SimVP Vis Kernel.')
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--fig_dir', default='./figure', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj  
    parser.add_argument('--hid_S', default=64, type=int)
    parser.add_argument('--hid_T', default=256, type=int)
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=51, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    return parser

class Vis:
    def __init__(self, args):
        super(Vis, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.res_dir = self.args.res_dir
        self.ex_name = self.args.ex_name
        self.fig_dir = self.args.fig_dir
        self.device = self._acquire_device()

        self._build_model()
        self._load_model()
        self._get_data()

     # CUDA Connection
    def _acquire_device(self):
        if self.args.use_gpu:
            # args.use_gpu가 True일 경우, CUDA에 args.gpu에 기재된 GPU Number를 입력.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
        else:
            device = torch.device('cpu')
        return device   
        
    # API/dataloader.py로부터 데이터 로드 : {MMNIST, taxibj}
    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    # model.py로부터
    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)

    # load saved model
    def _load_model(self):
        best_model_path = self.res_dir + '/' + self.ex_name + '/' + 'best_model.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    # 10 batch Sampler
    def _sampler(self):
        # 평가 모드 변환.
        self.model.eval()
        print(f'Sample Size : {self.args.val_batch_size} * 10 \n')   

        # 배치 별 Prediction & Output 저장.
        inputs_lst, trues_lst, preds_lst = [], [], []
        for i, (batch_x, batch_y) in enumerate(tqdm(self.test_loader)):
            
            # Stopper : i개의 Sample만 출력.
            if i == 5 : break
            
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))       
        
        print(f'batch_x: {batch_x.shape}, {torch.mean(batch_x)}')
        print(f'batch_y: {batch_y.shape}, {torch.mean(batch_y)}')
        print(f'pred_y: {pred_y.shape}, {torch.mean(pred_y)}')        

        # 총 결과 저장 및 Concat.
        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])
        
        return inputs, trues, preds
    
    # Visualize
    def visualize(self, args):
        # Sampling
        inputs, trues, preds = self._sampler()
        true_frame = np.concatenate([inputs,trues],axis=1)
        pred_frame = np.concatenate([inputs,preds],axis=1)
        print(true_frame.shape) # ------------------------------ 지울 것.
        print(pred_frame.shape) # ------------------------------ 지울 것.        
        
        idx = np.random.choice(inputs.shape[0],1)[0]
        print(f'SS idx : {idx}')

        # 20장의 Frame을 모두 시각화.
        vis_tool.multi_frame(true_frame[idx], path=(self.args.fig_dir + '/multi_f_true.jpg'), dataname='mmnist')
        vis_tool.multi_frame(pred_frame[idx], path=(self.args.fig_dir + '/multi_f_pred.jpg'), dataname='mmnist')

        # True값과 Pred값을 비교.
        vis_tool.comparison(true_frame[idx], pred_frame[idx], path=(self.args.fig_dir + '/comparison.jpg'), dataname='mmnist')

        # 단일 비디오 시각화.
        # single_true_video = vis_tool.create_single_video(true_frame[idx], path = (self.args.fig_dir + '/single_true_video.gif'))
        # single_pred_video = vis_tool.create_single_video(pred_frame[idx], path = (self.args.fig_dir + '/single_pred_video.gif'))

        # 다중 비디오 시각화.
        multi_true_video = vis_tool.create_multi_video(true_frame, 3, path = (self.args.fig_dir + '/multi_true_video.gif'))
        multi_pred_video = vis_tool.create_multi_video(pred_frame, 3, path = (self.args.fig_dir + '/multi_pred_video.gif'))

        #

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    exp = Vis(args)
    print('>>>>>>>>>>>>>>>>>>>>>>> Visualization <<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.visualize(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> End <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
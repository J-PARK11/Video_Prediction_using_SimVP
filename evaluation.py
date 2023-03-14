import os
import os.path as osp
import argparse
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from model import SimVP
import visualization
from API import *
from utils import *

'''
230309 Try1 : python evaluation.py --epochs=200 --res_dir='./results/230309_OG_SimVP' --fig_dir='./figure/230309_OG_SimVP' --batch_size=16 --val_batch_size=16 --dataname='mmnist'
230310 Try2 : python evaluation.py --epochs=1000 --res_dir='./results/230310_OG_SimVP_mmnist_1000' --fig_dir='./figure/230310_OG_SimVP_mmnist_1000' --batch_size=16 --val_batch_size=16 --dataname='mmnist'
230313 Try3 : python evaluation.py --epochs=60 --res_dir='./results/230313_OG_SimVP_kth_60' --fig_dir='./figure/230313_OG_SimVP_kth_60' --batch_size=16 --val_batch_size=16 --dataname='kth'
230314 Try4 : python evaluation.py --epochs=60 --res_dir='./results/230314_OG_SimVP_kth_60' --fig_dir='./figure/230314_OG_SimVP_kth_60' --batch_size=4 --val_batch_size=4 --dataname='kth'
230314 Try5 : python evaluation.py --epochs=10 --res_dir='./results/230314_OG_SimVP_kth_1000' --fig_dir='./figure/230314_OG_SimVP_kth_1000' --batch_size=8 --val_batch_size=4 --dataname='kth' --log_step=5
'''

def create_parser():
    parser = argparse.ArgumentParser(description='SimVP Eval Kernel.')
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
    parser.add_argument('--dataname', default='kth', choices=['mmnist', 'taxibj','kth'])
    parser.add_argument('--out_frame', default=10, type=int, help='Num of output frame')    
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 120, 160], type=int,nargs='*')
    # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj, [10, 1, 120, 160] for kth
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

class Eval:
    def __init__(self, args):
        super(Eval, self).__init__()
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
        
    # Evaluation
    def evaluation(self, args):
        # 평가 모드 변환.
        self.model.eval()
        print(f'Test Set Size : {len(self.test_loader)} * {self.args.val_batch_size}\n')            

        # 배치 별 Prediction & Output 저장.
        inputs_lst, trues_lst, preds_lst = [], [], []
        timechecker = TimeHistory('Evaluation')
        timechecker.begin()
        for batch_x, batch_y in tqdm(self.test_loader):
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))       
        timechecker.end()
        timechecker.print(local_print=True)

        print(f'batch_x: {batch_x.shape}, {torch.mean(batch_x)}')
        print(f'batch_y: {batch_y.shape}, {torch.mean(batch_y)}')
        print(f'pred_y: {pred_y.shape}, {torch.mean(pred_y)}')

        # 총 결과 저장 및 Concat.
        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        # Test 출력 저장 Path 설정.
        folder_path = self.res_dir + '/' + self.ex_name + '/outputs/{}/eval/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Test 결과 평가 지표 계산 및 출력.
        if self.args.dataname == 'kth':
            test_loader_mean = self.test_loader.dataset[0][0].mean().numpy()
            test_loader_std = self.test_loader.dataset[0][0].std().numpy()
        else:
            test_loader_mean = self.test_loader.dataset.mean
            test_loader_std = self.test_loader.dataset.std        
        mse, mae, ssim, psnr = metric(preds, trues, test_loader_mean, test_loader_std, True)
        print('test eval - mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        # Test 출력 결과 저장.
        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    exp = Eval(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluation <<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.evaluation(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> End <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

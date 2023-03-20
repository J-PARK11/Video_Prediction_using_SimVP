import argparse
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

import utils

'''
Terminal Sample
Debug : python main.py --epochs=10 --res_dir='./results/debug' --fig_dir='./figure/debug'
230309 Try1 : python main.py --epochs=200 --res_dir='./results/230309_OG_SimVP' --fig_dir='./figure/230309_OG_SimVP' --batch_size=16 --val_batch_size=16 --dataname='mmnist'
230310 Try2 : python main.py --epochs=1000 --res_dir='./results/230310_OG_SimVP_mmnist_1000' --fig_dir='./figure/230310_OG_SimVP_mmnist_1000' --batch_size=16 --val_batch_size=16 --dataname='mmnist'
230313 Try3 : python main.py --epochs=60 --res_dir='./results/230313_OG_SimVP_taxibj_60' --fig_dir='./figure/230313_OG_SimVP_taxibj_60' --batch_size=16 --val_batch_size=16 --dataname='taxibj'
230314 Try4 : python main.py --epochs=60 --res_dir='./results/230314_OG_SimVP_kth_60' --fig_dir='./figure/230314_OG_SimVP_kth_60' --batch_size=8 --val_batch_size=8 --dataname='kth'
230314 Try5 : python main.py --epochs=1000 --res_dir='./results/230314_OG_SimVP_kth_1000' --fig_dir='./figure/230314_OG_SimVP_kth_1000' --batch_size=8 --val_batch_size=4 --dataname='kth' --log_step=5
230317 Try6 : python main.py --epochs=100 --res_dir='./results/230317_OG_SimVP_caltech_100' --fig_dir='./figure/230317_OG_SimVP_caltech_100' --batch_size=8 --val_batch_size=4 --dataname='caltech' --log_step=1
230317 Try7 : python main.py --epochs=300 --res_dir='./results/230317_OG_SimVP_caltech_300' --fig_dir='./figure/230317_OG_SimVP_caltech_300' --batch_size=8 --val_batch_size=4 --dataname='caltech' --log_step=5
230317 Try8 : python main.py --epochs=300 --res_dir='./results/230317_OG_SimVP_caltech_6_10_300' --fig_dir='./figure/230317_OG_SimVP_caltech_6_10_300' --batch_size=8 --val_batch_size=4 --dataname='caltech' --log_step=5
230317 Try9 : python main.py --epochs=300 --res_dir='./results/230317_OG_SimVP_caltech_6_15_300' --fig_dir='./figure/230317_OG_SimVP_caltech_6_15_300' --batch_size=8 --val_batch_size=4 --dataname='caltech' --log_step=5
'''

def create_parser():
    parser = argparse.ArgumentParser(description='SimVP Train Kernel.')
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
    parser.add_argument('--val_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='caltech', choices=['mmnist', 'taxibj','kth','caltech'])
    parser.add_argument('--out_frame', default=10, type=int, help='Num of output frame')
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 3, 128, 160], type=int,nargs='*')
     # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj, [10, 1, 120, 160] for kth, [10, 3, 128, 160]
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

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)
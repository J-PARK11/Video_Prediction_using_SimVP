import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

from model import SimVP
from API import *
from utils import *

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._get_data()    # ???? self.preparation()에서 한 번 함.
        self._select_optimizer()
        self._select_criterion()
        self.early_stopping = EarlyStopping(patience=10, delta=0)

    # CUDA Connection
    def _acquire_device(self):
        if self.args.use_gpu:
            # args.use_gpu가 True일 경우, CUDA에 args.gpu에 기재된 GPU Number를 입력.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    # 각종 세팅.
    def _preparation(self):
        # seed
        set_seed(self.args.seed)

        # log and checkpoint : 학습 결과 폴더 및 명칭 정의.
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')   # checkpoints
        check_dir(self.checkpoints_path)

        self.tensorboard_path = osp.join(self.path, 'tensorboard')   # tensorboard

        sv_param = osp.join(self.path, 'model_param.json')   # model_parameters
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:           # messeage log path
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        
        # prepare data : 데이터 로드
        # self._get_data()
        # build the model : SimVP 모델 로드
        self._build_model()

    # model.py로부터
    def _build_model(self):
        args = self.args
        self.model = SimVP(tuple(args.in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(self.device)

    # API/dataloader.py로부터 데이터 로드 : {MMNIST, taxibj}
    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.train_mean, self.train_std, self.test_mean, self.test_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    # 옵티마이저 선택.
    def _select_optimizer(self):
        # 기본적으로 Adam 옵티마이저를 사용하되, 원사이클 스케줄러 사용.
        # 초기 Adam에 설정한 lr에서 max_lr까지 올라갔다가 다시 쭉 감소하는 스케줄러.
        # 현재 초기 lr과 max lr이 같은데, 추후에 변경 소지 있음.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    # 손실 함수 정의.
    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()

    # 저장 함수.
    def _save(self, name='SimVP'):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

    # Train
    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        save_model_config(self.model, self.optimizer, self.criterion, self.path+'/'+'model_config.txt')

        writer = SummaryWriter(self.tensorboard_path)
        timechecker = TimeHistory('Training')

        timechecker.begin()
        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                # 가중치 그레디언트 초기화.
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)

                loss = self.criterion(pred_y, batch_y)

                # 배치 단위 손실 값 저장
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                # 역전파 수행.
                loss.backward()
                # 옵티마이저 및 스케줄러 다음 스텝 진행.
                self.optimizer.step()
                self.scheduler.step()
                
            # 에포크 손실 값 = 배치 별 손실 값의 평균
            train_loss = np.average(train_loss)
            writer.add_scalar("Loss/train", train_loss, epoch)

            # log_step 당 중간 학습 결과 출력 및 저장.
            if epoch % args.log_step == 0:
                with torch.no_grad():
                    vali_loss = self.vali(self.vali_loader) 
                    if epoch % (args.log_step * 100) == 0:
                        self._save(name=str(epoch))

                print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                    epoch + 1, train_loss, vali_loss))
                recorder(vali_loss, self.model, self.path)
            writer.add_scalar("Loss/valid", vali_loss, epoch)

            # 조기종료 검증
            self.early_stopping(vali_loss)
            if self.early_stopping.is_stop(): 
                print_log("Early stopping")
                break

        timechecker.end()
        timechecker.print()


        # API/recoder에서 저장한 Best_model load
        best_model_path = self.path + '/' + 'best_model.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        # 모델의 평가 단계 돌입.
        self.model.eval()
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break
            
            # Valid data batch CUDA 연결
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            # Prediction
            pred_y = self.model(batch_x)
            # preds_lst, trues_lst에 CUDA 연결 해제 후 결과 넣기.
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            # loss 값 계산.
            loss = self.criterion(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            
            # 배치 별 로스 값 축적.
            total_loss.append(loss.mean().item())

        # 전체 Valid loss = 배치 별 축적된 손실 값의 평균.
        total_loss = np.average(total_loss)

        # Total preds, trues numpy concatenate
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)

        # 평가지표 계산 및 출력.
        if self.args.dataname in ['kth', 'caltech']:
            vali_loader_mean = self.test_mean
            vali_loader_std = self.test_std
        else:
            vali_loader_mean = self.vali_loader.dataset.mean
            vali_loader_std = self.vali_loader.dataset.std
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader_mean, vali_loader_std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        # 다시 훈련 모드로 변환.
        self.model.train()
        return total_loss

    def test(self, args):
        # 평가 모드 변환.
        self.model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []

        # 배치 별 Prediction & Output 저장.
        for batch_x, batch_y in self.test_loader:
            pred_y = self.model(batch_x.to(self.device))
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        # 총 결과 저장 및 Concat.
        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        # Test 출력 저장 Path 설정.
        folder_path = self.path+'/outputs/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Test 결과 평가 지표 계산 및 출력.
        if self.args.dataname in ['kth', 'caltech']:
            test_loader_mean = self.test_mean
            test_loader_std = self.test_std
        else:
            test_loader_mean = self.test_loader.dataset.mean
            test_loader_std = self.test_loader.dataset.std
        mse, mae, ssim, psnr = metric(preds, trues, test_loader_mean, test_loader_std, True)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        # Test 출력 결과 저장.
        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse
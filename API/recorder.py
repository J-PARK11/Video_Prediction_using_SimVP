import numpy as np
import torch

# 체크포인트 & Best Model Recorder
class Recorder:
    def __init__(self, verbose=False, delta=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        # 모든 체크포인트를 저장하는 경우.
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        
        # Best Score만 저장하는 경우.
        elif score >= self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)

    # 베스트 모델 저장.
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'best_model.pth')
        self.val_loss_min = val_loss
import os
import logging
import random
import time, datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# seet seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

# print both log file * prompt
def print_log(message):
    print(message)
    logging.info(message)

# argparse 파라미터 로깅.
def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message

# 폴더 존재 유무 체크 및 생성.
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# custom ------------------------------------------------------------#
# Tensorboard directory path definition
def make_tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_logdir, sub_dir_name)

# Model Time Consumption Checker
class TimeHistory():
    def __init__(self, name):
        self.start_time = 0
        self.end_time = 0
        self.name = name
    
    def begin(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
    
    def print(self, local_print=False):
        if ((self.start_time > 0) and (self.end_time > 0)):
            sec = self.end_time - self.start_time
            result = datetime.timedelta(seconds=sec)
            result = str(result).split('.')[0]
            
            if local_print : print(f'{self.name} : {result}')
            else: print_log(f'{self.name} : {result}')

# Save Model config
def save_model_config(model, optim, loss, path):
    f = open(path, 'w')
    
    f.write('Models Children:\n')
    for child in model.children():
        f.write('===')
        f.write(f'{child}')
    
    # f.write('\n\n')
    # f.write("Model's state_dict:\n")
    # for param_tensor in model.state_dict():
    #     f.write(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # f.write('\n\n')
    # f.write('Models named modules')
    # for name,layer in model.named_modules():
    #     f.write(name,layer)

    # f.write('\n\n')
    # f.write("Optimizer's state_dict:")
    # for var_name in optim.state_dict():
    #     f.write(var_name, "\t", optim.state_dict()[var_name])

    # f.write('\n\n')
    # f.write("loss state_dict:")
    # for var_name in loss.state_dict():
    #     f.write(var_name, "\t", loss.state_dict()[var_name])

    f.close()
# --------------------------------------------------------------------#

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.loss = np.inf
        self.patience = 0
        self.limit = patience
        self.delta = delta
        self.stop = False
    
    def __call__(self, loss):
        # 성능이 향상될 경우.
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        # 성능이 향상되지 않을 경우.
        else:
            self.patience += 1

        # Patience limit 확인.
        if self.patience > self.limit : self.stop = True

    def is_stop(self):
        return self.stop
    

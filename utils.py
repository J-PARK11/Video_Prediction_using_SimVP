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
# --------------------------------------------------------------------#
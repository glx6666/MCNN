import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import random


# 初始化参数
def init_seed(seed=123):
    seed = seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True

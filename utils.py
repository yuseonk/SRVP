import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import shutil
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import logging
import random
import gc
import time
import copy
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Arial'
import seaborn as sns
plt.style.use('seaborn')
sns.set(style='white', context='notebook', palette="muted", color_codes=True, font_scale=2)
from matplotlib import cm
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms, utils
from torch import nn, autograd
from torch.autograd import Variable
from torch.nn import MSELoss, L1Loss, BCELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
Tensor = torch.FloatTensor
cuda = True if torch.cuda.is_available() else False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # cpu
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # gpu 
        torch.cuda.manual_seed_all(seed)  # multi-gpu
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class ToTensor(torch.nn.Module):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        sample = np.transpose(sample, (0, 3, 1, 2))  # T x C x H x W
        sample = (sample - sample.min()) / (sample.max() - sample.min())  # 0-1
        return torch.from_numpy(sample)
    

class VideoDataset(Dataset):
    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        input_seq = self.datasets[idx][0]
        label_seq = self.datasets[idx][1]

        if len(input_seq.shape) < 4:
            input_seq = np.expand_dims(input_seq, -1)
        if len(label_seq.shape) < 4:
            label_seq = np.expand_dims(label_seq, -1)

        if self.transform:
            input_seq = self.transform(input_seq)
        if self.target_transform:
            label_seq = self.target_transform(label_seq)
        return {'input': input_seq, 'label': label_seq}
    

class Estimator():
    def __init__(self, epoch=None, bs=None, net=None, train_set=None, val_set=None, ctx=0, dst_path='./'):
        self.num_epochs = epoch
        self.batch_size = bs
        self.train_set = train_set
        self.val_set = val_set
        self.net = net
        self.train_loss, self.val_loss = [], []
        self.best_loss = float("Inf")
        self.dst_path = dst_path
        self.device = torch.device(f'cuda:{ctx}' if cuda else 'cpu')
        self.scheduler = None

    def init_net(self, lr):
        self.base_lr = lr
        self.loss_fn = BCELoss()
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.base_lr, weight_decay=1e-4, momentum=0.9)
        num_iter = len(self.train_set) // self.batch_size
        tmax = num_iter*self.num_epochs
        self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                        T_max=tmax,
                                                        eta_min=1e-7)
            
    def training(self, logger=None):
        total_loss = torch.zeros(1, device=self.device)
        self.net.train()

        for i, data in enumerate(self.train_set):
            inputs, label = data['input'].float().to(self.device), data['label'].float().to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                pred = self.net(inputs)
                losses = self.loss_fn(pred, label)

            losses.backward()
            self.optimizer.step()
            # loss, metric update
            total_loss += losses.item()

        self.scheduler.step()  # (option)
        total_loss = total_loss / len(self.train_set)
        self.train_loss.append(total_loss.item())
        logger.info("Train loss: {:.6f}".format(self.train_loss[-1]))

    def validation(self, logger=None):
        total_loss = torch.zeros(1, device=self.device)
        self.net.eval()

        for i, data in enumerate(self.val_set):
            inputs, label = data['input'].float().to(self.device), data['label'].float().to(self.device)

            with torch.set_grad_enabled(False):
                pred = self.net(inputs)
                losses = self.loss_fn(pred, label)
            total_loss += losses.item()

        total_loss = total_loss / len(self.val_set)
        self.val_loss.append(total_loss.item())
        logger.info("Valid loss: {:.6f}".format(self.val_loss[-1]))

    def do(self, logger=None, lr=0.01):
        tic = time.time()

        self.init_net(lr)
        for epoch in range(self.num_epochs):
            etic = time.time()
            logger.info('Epoch {}'.format(epoch + 1))

            self.training(logger)
            self.validation(logger)
            gc.collect()

            logger.info('Time: {:.3f}'.format(time.time() - etic))
            if np.isnan(self.val_loss[-1]):
                break
            if self.val_loss[-1] < self.best_loss:
                self.best_loss = self.val_loss[-1]
                torch.save(self.net.state_dict(), f'{self.dst_path}/best_loss.pt')
                logger.info('save best loss model')

            loss_results = pd.DataFrame(data={'Train': self.train_loss, 'Valid': self.val_loss})
            loss_results.to_csv(os.path.join(self.dst_path, 'loss_results.csv'), index=False)
            torch.save(self.net.state_dict(), f'{self.dst_path}/model.pt')
            torch.save(self.scheduler.state_dict(), f'{self.dst_path}/checkpoint.pt')
        logger.info('Train speed: {:.3f}, Best loss of validset: {}'.format(time.time() - tic, self.best_loss))


def inverse_scale(img, max_=255., min_=0.):
    img_ = img * (max_ - min_) + min_
    return np.round(img_)


def float_to_uint(arr):
    return arr.astype(np.uint8)


class Evaluator():
    def __init__(self, net=None, test_set=None, ctx=0, idx=None):
        self.net = net
        self.test_set = test_set
        self.device = torch.device(f'cuda:{ctx}' if cuda else 'cpu')
        self.idx = idx

    def postprocessing(self, arr):
        return float_to_uint(inverse_scale(np.transpose(arr, (0, 1, 3, 4, 2))))

    def inference(self, model_path):
        gc.collect()
        self.net.load_state_dict(torch.load(f'{model_path}/best_loss.pt'))
        self.net.to(self.device)
        self.net.eval()

        pred_result = []
        label_result = []
        for data in self.test_set:
            inputs, label = data['input'].float().to(self.device), data['label'].float().to(self.device)
            
            with torch.set_grad_enabled(False):
                pred = self.net(inputs)
            pred_result.append(pred.cpu().detach())
            label_result.append(label.cpu().detach())

        preds = np.concatenate([batch.numpy() for batch in pred_result])
        labels = np.concatenate([batch.numpy() for batch in label_result])

        return self.postprocessing(preds), self.postprocessing(labels)


def get_score(prediction, groundtruth, horizon=1):
    # pixel-wise compare
    mse = mean_squared_error(groundtruth, prediction)
    psnr = peak_signal_noise_ratio(groundtruth, prediction, data_range=prediction.max() - prediction.min())
    channel_axis = -1
    # image structure-based compare
    ssim = structural_similarity(groundtruth, prediction,
                                 channel_axis=channel_axis,
                                 data_range=prediction.max() - prediction.min())

    return pd.DataFrame([mse, psnr, ssim], index=['MSE', 'PSNR', 'SSIM'],
                        columns=[f'h{horizon}']).T

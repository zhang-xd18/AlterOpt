import time
import os
import torch
from collections import namedtuple
from utils import logger
from utils.func import set_model_
from utils.statics import AverageMeter, evaluator
from copy import deepcopy

__all__ = ['Alter']


field = ('nmse', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Alter:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, scheduler, save_path='./checkpoint', print_freq=20, test_freq=1):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.add_optimizer = None
        self.scheduler = scheduler
        self.add_scheduler = None
        self.add_train = False
        self.device = device
        
        # control parameters importances
        self.mse = torch.nn.MSELoss()
        self.best_model = None
        self.best_epoch = None
        self.best_epoch = None

        # Verbose arguments
        self.save_path = save_path
        self.print_freq = print_freq
        self.test_freq = test_freq

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.test_loss = None

        self.test_loader = None
        self.test_nmse = []

    def loop(self, t, epochs, train_loader, test_loader, pretrain=None, add_train=False):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            test_loader (DataLoader): Data loader for test data.
        """
        self._pretrain(pretrain)
        self.add_train = add_train
        
        self.cur_epoch = 1
        self.best_nmse = None
        self.all_epoch = epochs
        
        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep 

            # conduct training, validation and test
            self.train_loss = self.train(t, train_loader)
            
            if ep % self.test_freq == 0:
                nmse_list = self.test(t, test_loader)
                
            self._epoch_postprocessing(t, nmse_list)
        
        # restore and retest the best
        set_model_(self.model, self.best_model)
        nmse_list = self.test(t, test_loader)
        self._loop_postprocessing(t)
        
        return nmse_list

    def add_optim(self, optim, scheduler):
        self.add_optimizer = optim
        self.add_scheduler = scheduler
        return 

    def train_criterion(self, t, sparse_gt, sparse_pred):
        return self.mse(sparse_gt, sparse_pred)
    
    def train(self, t, train_loader):
        r""" train the model on the given data loader for one epoch.

        Args:
            train_loader (DataLoader): the training data loader
        """

        self.model.train()
        with torch.enable_grad():
            return self._iteration(t, train_loader)

    def test(self, t, test_loader):
        r""" Truly test the model on the test dataset for one epoch.

        Args:
            test_loader (DataLoader): the test data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self._iteration_test(t, test_loader, self.train_criterion)

    def _iteration_test(self, t, data_loader, criterion):
        r""" protected function which test the model on given data loader for one epoch.
        """
        n = len(data_loader)
        iter_data_list = [iter(data_loader[_]) for _ in range(n)]
        iter_nmse_list = [AverageMeter(f'Iter nmse_{_}') for _ in range(n)]
        iter_loss_list = [AverageMeter(f'Iter loss_{_}') for _ in range(n)]
        iter_mse_list = [AverageMeter(f'Iter mse_{_}') for _ in range(n)]
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx in range(len(data_loader[0])):
            for nn in range(n):
                sparse_gt = next(iter_data_list[nn])[0].to(self.device)
                sparse_pred = self.model(sparse_gt)
                loss = criterion(t, sparse_pred, sparse_gt)
                mse = self.mse(sparse_pred, sparse_gt)
                nmse = evaluator(sparse_pred, sparse_gt)
                iter_loss_list[nn].update(loss)
                iter_nmse_list[nn].update(nmse)  
                iter_mse_list[nn].update(mse) 
        
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader[0])}] ')
                for nn in range(n):
                    logger.info(f'loss_{nn}: {iter_loss_list[nn].avg:.3e} | ' 
                                f'MSE_{nn}: {iter_mse_list[nn].avg:.3e} | '
                                f'NMSE_{nn}: {iter_nmse_list[nn].avg:.3e} ')
                logger.info(f'time: {iter_time.avg:.3f}')
        
        for nn in range(n):
            logger.info(f'Test NMSE_{nn}: {iter_nmse_list[nn].avg:.3e}')
        return [iter_nmse_list[nn].avg for nn in range(n)]

    def _iteration(self, t, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        # need to update for normalization 
        for batch_idx, (sparse_gt, ) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.train_criterion(t, sparse_pred, sparse_gt)

            # Scheduler update, backward pass and optimization
            if self.model.training:
                if self.add_train == True:
                    self.add_optimizer.zero_grad()
                    loss.backward()
                    self.add_optimizer.step()
                    self.add_scheduler.step()
                    lr = self.add_scheduler.get_lr()[0]
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    lr = self.scheduler.get_lr()[0]

            # Log and visdom update
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {lr:.2e} | '
                            f'MSE loss: {iter_loss.avg:.3e} | '
                            f'time: {iter_time.avg:.3f}')

        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> {mode}  Loss: {iter_loss.avg:.3e}\n')

        return iter_loss.avg

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return
        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))
        
    def _pretrain(self, pretrain=None):
        r""" protected function which resume from checkpoint at the beginning of training.
        """
        if pretrain is not None:
            assert os.path.isfile(pretrain)
            logger.info(f'=> loading checkpoint {pretrain}')
            checkpoint = torch.load(pretrain)
            self.model.load_state_dict(checkpoint['state_dict'])
            logger.info(f'=> successfully loaded checkpoint {pretrain}\n')
        return     
    
    def _epoch_postprocessing(self, t, nmse_list):
        r""" private function which makes loop() function neater.
        """
        # choose the best nmse on the current task as the criterion
        if nmse_list is not None:
            if self.add_train:
                nmse = torch.mean(nmse_list[-1] + nmse_list[-2])
            else:
                nmse = nmse_list[-1]
            if self.best_nmse is None or self.best_nmse > nmse:
                self.best_nmse = nmse
                self.best_epoch = self.cur_epoch
                self.best_model = deepcopy(self.model.state_dict())
            
            if self.best_nmse is not None:
                logger.info(f'\n   Best NMSE on task {t}: {self.best_nmse:.3e} '
                            f'\n                   epoch: {self.best_epoch} \n')               
        return 
    
    def _loop_postprocessing(self, t):
        state = {
            'best_epoch': self.best_epoch,
            'best_nmse': self.best_nmse,
            'state_dict': self.model.state_dict(),
        }
        self._save(state, name=f'best_after_{t}.pth')
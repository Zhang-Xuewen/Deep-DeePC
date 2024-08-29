"""
Name: epoch.py
Author: Xuewen Zhang
Date: 19/04/2024
Project: DeepDeePC
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from .model import network
from QYtool import rprint, datatool, dirtool, mathtool, timer, progressbar
# torch.autograd.set_detect_anomaly(True)


import threading
from functools import wraps

def run_in_thread(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

class train(object):
    def __init__(self, args):
        # system parameters
        self.system = args['system']
        self.u_dim = args['u_dim']
        self.y_dim = args['y_dim']
        self.p_dim = args['p_dim']
        self.x0_name = args['x0_name']
        self.p_name = args['p_name']
        self.sp_dir = args['sp_dir']
        self.sys_param_dir = args['sys_param_dir']
        self.y_idx = args['y_idx']
        self.x0_std_dev = args['x0_std_dev']
        self.noise = args['noise']
        
        # deepc parameters
        self.T = args['T']
        self.Tini = args['Tini']
        self.Np = args['Np']
        self.N = args['N']
        self.sp_num = args['sp_num']
        self.RDeePC = args['RDeePC']
        
        # learning parameters
        self.lr = args['lr']
        self.epoch = args['epoch']
        self.model_dir = args['model_dir']
        self.data_dir = args['data_dir']
        self.fig_dir = args['fig_dir']
        self.batch_size = args['batch_size']
        self.data_size = args['data_size']
        
        self.model = network(args['input_size'], args['output_size'], args['hidden_size_list'])
        
        # load the plant
        xs, us = datatool.loadtxt(self.x0_name, 'xs', 'us')
        if self.system == 'WWTPs':
            from .Plants.waste_water_system import waste_water_system as simulator
            x0, p = datatool.loadtxt(f'./_data/{system}/', 'ss_open', self.p_name) 
            self.plant = simulator(x0, p, x0_std_dev=self.x0_std_dev, xs=xs, us=us)
        elif self.system == 'threetanks':
            from .Plants.three_tanks import three_tank_system as simulator
            self.plant = simulator(x0_std_dev=self.x0_std_dev, xs=xs, us=us)
        elif self.system == 'siso':
            from .Plants.siso import siso as simulator
            self.plant = simulator(x0_std_dev=self.x0_std_dev, xs=xs, us=us)
        elif self.system == 'grn':
            from .Plants.grn import grn as simulator
            self.plant = simulator()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.mseloss = nn.MSELoss()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('-'*15, f"{'GPU' if torch.cuda.is_available() else 'CPU'}", "-"*15)
        self.model.to(self.device)

        self._data_process(args) 
        
        # Put all the tensors to the specified device
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(self.device))
        
    
    def _data_process(self, args):
        # load offline collected data for DeePC
        self.offline_dir = args['offline_dir']
        ud, yd = datatool.loadtxt(self.offline_dir, 'u', 'y')
        ud = ud.reshape(-1, self.u_dim)
        yd = yd.reshape(-1, self.y_dim)
        
        # load the open-loop data for training the neural network
        self.openloop_dir = args['openloop_dir']
        u, y = datatool.loadtxt(self.openloop_dir, 'u', 'y')
        u = u.reshape(-1, self.u_dim)
        y = y.reshape(-1, self.y_dim)
        
        # scale the data
        self.scale_dir = args['scale_dir']
        u_sc1, u_sc2 = datatool.loadtxt(self.scale_dir, 'u_min', 'u_max')
        y_sc1, y_sc2 = datatool.loadtxt(self.scale_dir, 'y_min', 'y_max')
        
        self.u_sc1, self.u_sc2, self.y_sc1, self.y_sc2 = \
            u_sc1.reshape(1, -1), u_sc2.reshape(1, -1), y_sc1.reshape(1, -1), y_sc2.reshape(1, -1)
        
        self.u_sc1_torch, self.u_sc2_torch = datatool.toTorch(self.u_sc1, self.u_sc2)
        self.y_sc1_torch, self.y_sc2_torch = datatool.toTorch(self.y_sc1, self.y_sc2)
        
        ud_, u_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', ud, u)   
        yd_, y_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', yd, y)     
        
        Hlu_ = mathtool.hankel(self.Tini+self.Np, 'tensor', torch.FloatTensor(ud_[:self.T, :]))
        Hly_ = mathtool.hankel(self.Tini+self.Np, 'tensor', torch.FloatTensor(yd_[:self.T, :]))
        
        self.Up_, self.Uf_ = Hlu_[:self.u_dim*self.Tini, :].clone(), Hlu_[self.u_dim*self.Tini:, :].clone()
        self.Yp_, self.Yf_ = Hly_[:self.y_dim*self.Tini, :].clone(), Hly_[self.y_dim*self.Tini:, :].clone()
        
        # make the input and label of the neural network
        u_ = u_[:self.data_size, :]
        y_ = y_[:self.data_size, :]
        L = self.Tini + self.Np
        u_L_ = torch.FloatTensor(np.array([u_[i:i+L, :] for i in range(u_.shape[0]-L+1)])).to(self.device)  # (ud_.shape[0]-L+1, L, u_dim)
        y_L_ = torch.FloatTensor(np.array([y_[i:i+L, :] for i in range(y_.shape[0]-L+1)])).to(self.device)  # (yd_.shape[0]-L+1, L, y_dim)
        
        input_data, label_data = [], []
        for i in range(len(u_L_)):
            uini_ = u_L_[i, :self.Tini, :].flatten()
            yini_ = y_L_[i, :self.Tini, :].flatten()
            # u_e_ = u_L_[i, -1, :].flatten() - u_L_[i, self.Tini-1, :].flatten()   # TODO: u_e_ = u_{k+Np-1}^r - u_{k-1} or u_{k}^r - u_{k-1}
            # y_e_ = y_L_[i, -1, :].flatten() - y_L_[i, self.Tini-1, :].flatten()
            u_e_ = u_L_[i, self.Tini, :].flatten() - u_L_[i, self.Tini-1, :].flatten()
            # y_e_ = y_L_[i, self.Tini, :].flatten() - y_L_[i, self.Tini-1, :].flatten()
            y_e_ = y_L_[i, self.Tini+1, :].flatten() - y_L_[i, self.Tini, :].flatten()
            input_part = torch.cat((uini_, yini_, u_e_, y_e_), dim=0)   # (Tini*(y_dim+u_dim),)
            label_part = torch.cat((u_L_[i, self.Tini:, :].flatten(), y_L_[i, self.Tini:, :].flatten()), dim=0)   # (Np*(y_dim+u_dim),)
            input_data.append(input_part)
            label_data.append(label_part)
        input_data = torch.stack(input_data)   # (ud.shape[0]-L+1, Tini*(y_dim+u_dim))
        label_data = torch.stack(label_data)   # (ud.shape[0]-L+1, Np*(y_dim+u_dim))
        
        self.dataloader = DataLoader(TensorDataset(input_data, label_data), batch_size=self.batch_size, shuffle=True)
        self.data_size = u_.shape[0] if u_.shape[0] < self.data_size else self.data_size
        self.batch_num = len(self.dataloader)
        
        # weighting matrices
        self.Q = torch.diag(torch.FloatTensor(args['Q']).repeat(self.Np)) 
        self.R = torch.diag(torch.FloatTensor(args['R']).repeat(self.Np)) 
        self.lambda_y = torch.diag(torch.FloatTensor(args['lambda_y']).repeat(self.Tini))
        self.lambda_g = args['lambda_g'] * torch.eye(self.T-self.Tini-self.Np+1)
        
        # bounds of the control input and controlled output
        u_lb = torch.FloatTensor(self.plant.action_low).view(1, -1)
        u_ub = torch.FloatTensor(self.plant.action_high).view(1, -1)
        y_lb = torch.FloatTensor(self.plant.y_low).view(1, -1)
        y_ub = torch.FloatTensor(self.plant.y_high).view(1, -1)
        
        u_lb_, u_ub_ = datatool.scale(self.u_sc1_torch, self.u_sc2_torch, 'minmax', 'tensor', u_lb, u_ub)
        y_lb_, y_ub_ = datatool.scale(self.y_sc1_torch, self.y_sc2_torch, 'minmax', 'tensor', y_lb, y_ub)
        
        self.u_lb_Np_ = torch.tile(u_lb_, (self.Np, 1)).view(-1, 1)
        self.u_ub_Np_ = torch.tile(u_ub_, (self.Np, 1)).view(-1, 1)
        self.y_lb_Np_ = torch.tile(y_lb_, (self.Np, 1)).view(-1, 1)
        self.y_ub_Np_ = torch.tile(y_ub_, (self.Np, 1)).view(-1, 1)
        
        self.P_u = torch.FloatTensor(np.tile(args['P_u'], self.Np)).to(self.device)
        self.P_y = torch.FloatTensor(np.tile(args['P_y'], self.Np)).to(self.device)
        return 
      
    
    def _sp_process(self):
        uref_all_, yref_all_ = [], []
        us_all_, ys_all_ = [], []
        us_all, ys_all = [], []
        sp_idx = np.arange(1, self.sp_num+1)
        # shuffle the set-points so that the trajectories changes each epoch
        np.random.shuffle(sp_idx)
        j = 0
        for i in sp_idx:
            j += 1
            us, ys = datatool.loadtxt(self.sp_dir+f'{i}/', 'us', 'ys')
            if j == 1:
                us0, ys0 = us, ys
            us_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', us)
            ys_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', ys)
            us_all.append(np.tile(us, (self.N if j != 1 else self.N + self.Tini, 1)))
            ys_all.append(np.tile(ys, (self.N if j != 1 else self.N + self.Tini, 1)))
            us_all_.append(np.tile(us_, (self.N if j != 1 else self.N + self.Tini, 1)))
            ys_all_.append(np.tile(ys_, (self.N if j != 1 else self.N + self.Tini, 1)))
            uref_all_.append(np.tile(us_, (self.N if j != 1 else self.N + self.Tini, 1)).reshape(-1, 1))
            yref_all_.append(np.tile(ys_, (self.N if j != 1 else self.N + self.Tini, 1)).reshape(-1, 1))
        uref_all_ = np.concatenate(uref_all_, axis=0)
        yref_all_ = np.concatenate(yref_all_, axis=0)
        us_all = np.concatenate(us_all, axis=0)
        ys_all = np.concatenate(ys_all, axis=0)
        us_all_ = np.concatenate(us_all_, axis=0)
        ys_all_ = np.concatenate(ys_all_, axis=0)
        us_all_, ys_all_, uref_all_, yref_all_ = datatool.toTorch(us_all_, ys_all_, uref_all_, yref_all_)
        us_all_, ys_all_, uref_all_, yref_all_ = datatool.toDevice(self.device, us_all_, ys_all_, uref_all_, yref_all_)
        return us_all, ys_all, us_all_, ys_all_, uref_all_, yref_all_
      
        
    def _data_to_step(self, x, t=None):
        """
            data: (data_size, x_dim)
            t: (data_size)
        """

        x = x.T
        dim, n = x.shape
        x_step = np.zeros((dim, 2*n))

        for i in range(n):
            x_step[:, 2 * i] = x[:, i]
            x_step[:, 2 * i + 1] = x[:, i]

        if t is not None:
            t_step = np.zeros(2*len(t))
            for i in range(len(t)):
                t_step[2 * i] = t[i]
                t_step[2 * i + 1] = t[i]
            ts = t[1]-t[0]
            t_step = t_step[1:]
            t_step = np.concatenate((t_step, np.array([t_step[-1] + ts])))
            return x_step.T, t_step
        else:
            return x_step.T
        
    @run_in_thread
    def _yu_plot(self, Y, U, us_all, ys_all, loss=None, epoch_num=None):
        Y = Y.clone()
        U = U.clone()
        
        ypath = np.vstack((self.yini0, Y.detach().numpy()))
        upath = np.vstack((self.uini0, U.detach().numpy()))
        t = np.arange(self.N * self.sp_num + self.Tini)*self.plant.sampling_period
        t_step, [upath_step, us_step, ys_step] = datatool.data_to_step(t, upath, us_all, ys_all)
        
        fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(20, 8))

        for i in range(3):
            ax[0, i].plot(t, ypath[:, i], '-')
            ax[0, i].axvline(x=self.Tini*self.plant.sampling_period, linestyle='--', color='b', label='Tini')
            ax[0, i].plot(t_step, ys_step[:, i], linestyle='--', color='g', label='Set-point')
            ax[0, i].set_xlabel('Time (day)')
            ax[0, i].set_ylabel(f"y{i+1}")
            
        for i in range(3):
            ax[1, i].plot(t_step, upath_step[:, i], '-')
            ax[1, i].axvline(x=self.Tini*self.plant.sampling_period, linestyle='--', color='b', label='Tini')
            ax[1, i].plot(t_step, us_step[:, i], linestyle='--', color='g', label='Set-point')
            ax[1, i].set_xlabel('Time (day)')
            ax[1, i].set_ylabel(f"u{i+1}")
        ax[1, i].legend()
        fig.suptitle(f"Epoch: {epoch_num+1}, loss: {loss}")
        fig.savefig(self.fig_dir + f"/{epoch_num+1}_yu.pdf")
        plt.close()
    
    
    def compute_loss(self, g_batch, inputs, label):
        """ Compute the objective function of the DDeePC 
        Args:
            g_batch (torch.Tensor): (batch_size, g_dim)
            inputs (torch.Tensor): (batch_size, (u_dim + y_dim)*(Tini+1))  [uini, yini, eu, ey]
            label (torch.Tensor): (batch_size, (u_dim + y_dim)*Np)
        """
        loss_dpc, loss_reg, loss_robust = 0, 0, 0
        for i in range(g_batch.shape[0]):
            g = g_batch[i, :].view(-1, 1)
            u_ = self.Uf_ @ g 
            y_ = self.Yf_ @ g
            uref_ = label[i, :self.u_dim*self.Np].view(-1, 1)
            yref_ = label[i, self.u_dim*self.Np:].view(-1, 1)
            
            loss_dpc += (yref_ - y_.clone()).T @ self.Q @ (yref_ - y_.clone()) + (uref_ - u_.clone()).T @ self.R @ (uref_ - u_.clone())
            loss_reg += self._loss_reg(u_, y_)
            if self.RDeePC:
                yini_ = inputs[i, self.u_dim*self.Tini:-(self.u_dim + self.y_dim)].view(-1, 1)
                loss_robust += self._loss_robust(g, yini_)          
        loss = (loss_dpc + loss_reg + loss_robust)/g_batch.shape[0]
        return loss
        
        
    def _loss_reg(self, u_, y_):
        """ Regularization loss for the bound constraints """
        loss_u_reg = torch.sum((u_ < self.u_lb_Np_) * self.P_u * (u_ - self.u_lb_Np_)**2) \
            + torch.sum((u_ > self.u_ub_Np_) * self.P_u * (u_ - self.u_ub_Np_)**2)
        loss_y_reg = torch.sum((y_ < self.y_lb_Np_) * self.P_y * (y_ - self.y_lb_Np_)**2) \
            + torch.sum((y_ > self.y_ub_Np_) * self.P_y * (y_ - self.y_ub_Np_)**2)
        # loss_reg = torch.exp(0.5 * loss_u_reg + 0.5 * loss_y_reg)
        # loss_reg = 10 * (loss_u_reg + loss_y_reg)
        loss_reg = loss_u_reg + loss_y_reg
        return loss_reg
    
    
    def _loss_robust(self, g, yini_):
        """ Robustness loss for the robust DeePC """
        # yini_ = self.yini_.clone().detach().view(-1, 1)   # this is detach the same as the inputs to the NN model is detached, assume that each iteration is new, just different uini, yini. To check the current loss
        loss_robust = (self.Yp_ @ g - yini_).T @ self.lambda_y @ (self.Yp_ @ g - yini_) + g.T @ self.lambda_g @ g 
        return loss_robust
        
        
    def _update_ini(self, uk_=None, yk_=None, us_=None, ys_=None):
        """
        Args:
            uk_ (_type_, optional): Control inputs of current instant. Defaults to None.
            yk_ (_type_, optional): Controlled outputs of current instant. Defaults to None.

        Returns:
            input: (1, (u_dim + y_dim)*Tini), input to the neural network
        """
        if uk_ is not None and yk_ is not None:
            self.uini_ = torch.cat((self.uini_[1:, :].clone(), uk_.clone().view(1, -1)), dim=0)
            self.yini_ = torch.cat((self.yini_[1:, :].clone(), yk_.clone().view(1, -1)), dim=0)
        else:
            self.uini_, self.yini_ = self.uini0_.clone(), self.yini0_.clone()  # reset each epoch
        uini_, yini_ = self.uini_.clone().view(1, -1), self.yini_.clone().view(1, -1)
        us_, ys_ = us_.view(1, -1), ys_.view(1, -1) 
        u_e_ = us_ - self.uini_[-1, :].clone()
        y_e_ = ys_ - self.yini_[-1, :].clone()
        inputs = torch.cat((uini_, yini_, u_e_, y_e_), dim=1)
        # inputs = torch.cat((uini_, yini_, y_e_), dim=1)
        return inputs
        
           
    def runepoch(self): 
        """ 
        Run the training process for each epoch 
        (one entire online training process with different set-points and initial conditions)
        """
        data_iter = iter(self.dataloader)
        loss_memory = []
        # Batch loop
        for i in range(self.batch_num):
            inputs, label = next(data_iter)
            
            g_batch = self.model(inputs)  # (batch_size, g_dim)
            
            loss = self.compute_loss(g_batch, inputs, label)
            
            # loss = self.mseloss(pred, label)
            loss_memory.append(loss.cpu().detach())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        loss_ave = torch.mean(torch.stack(loss_memory))
        return loss_ave
    
    
    @timer
    def run(self):
        """ Run the training process """
        loss_memory = []
        probar = progressbar.probar2()
        with probar:
            task = probar.add_task('Training', total=self.epoch, unit='epoch', loss=np.inf)
            rprint(f":gear: [green]Epoch: {self.epoch}, Batch number: {self.batch_num}, Batch size: {self.batch_size}, Data size: {self.data_size}[/green]")
            for i in range(self.epoch):
                self.model.train()
                loss_ave = self.runepoch()
                loss_memory.append(loss_ave)
                
                datatool.save_model(self.model, self.model_dir, loss_ave, loss_memory) 
                
                probar.update(task, advance=1, loss=loss_ave)
                datatool.savetxt(self.data_dir, loss_memory=loss_memory)
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(loss_memory, label='Training loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        fig.savefig(self.fig_dir + f"/training_loss.pdf")
        plt.close()
        
            
            
            
            
            
            
            
            
        
        
        
        
        
        

        

"""
Name: control.py
Author: Xuewen Zhang
Date: 23/04/2024
Project: DeepDeePC
"""
import os
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .deepc import deepc
from .model import network
from .con_opt import con_opt
# from .Plants.waste_water_system import waste_water_system as simulator
from .Plants.three_tanks import three_tank_system as simulator
from QYtool import rprint, datatool, dirtool, progressbar, timer, nntool, mathtool



class control(object):
    def __init__(self, args):
        self.exp_dir = args['exp_dir']
        self.fig_dir = args['control_fig_dir']
        self.open_loop = args['open_loop']
        
        # system parameters
        self.system = args['system']
        self.u_dim = args['u_dim']
        self.y_dim = args['y_dim']
        self.p_dim = args['p_dim']
        self.x0_name = args['x0_name']
        self.p_name = args['p_name']
        self.sys_param_dir = args['sys_param_dir']
        self.y_idx = args['y_idx']
        self.noise = args['noise']
        self.x0_std_dev = args['x0_std_dev']
        
        # deepc parameters
        self.T = args['T']
        self.Tini = args['Tini']
        self.Np = args['Np']
        self.N = args['N']
        self.g_dim = args['output_size']
        self.dpc_opts = args['dpc_opts']
        self.solver = args['solver']
        self.con_opt = args['con_opt']
        
        # load model, plant, deepc controller
        model = network(args['input_size'], args['output_size'], args['hidden_size_list'])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = nntool.load_model(self.exp_dir+'/model/train_model.pt', self.device, model, 'nn_model')
        self.model.to(self.device)
        self.model.eval()
        
        self.deepc = deepc(**args)
        # self.deepc.test()  # test the current config of deepc using 2 set-points
        
        # data process
        self._data_process(args)
        
        # initial constraint optimization solver
        if self.con_opt:
            self.con_solver = self._init_con_opt()
        return 
    
        
    def _data_process(self, args):
        # make control directory
        self.control_dir = args['control_dir']
        self.control_data_dir = self.control_dir + 'con_opt/' if self.con_opt else self.control_dir + 'no_con_opt/'
        dirtool.makedir(self.control_data_dir)
        
        # scale data
        self.scale_dir = args['scale_dir']
        self.u_sc1, self.u_sc2 = datatool.loadtxt(self.scale_dir, 'u_min', 'u_max')
        self.y_sc1, self.y_sc2 = datatool.loadtxt(self.scale_dir, 'y_min', 'y_max')
        # self.p_sc1, self.p_sc2 = datatool.loadtxt(self.scale_dir, 'p_min', 'p_max')
        
        # set point
        self.control_sp = args['control_sp']
        us_all, ys_all, us_all_, ys_all_ = [], [], [], []
        for i in range(len(self.control_sp)):
            us, ys = datatool.loadtxt(args['sp_dir'] + f'{self.control_sp[i]}/', 'us', 'ys')
            if i == 0:
                us0, ys0 = us.copy(), ys.copy()
                xs0 = np.loadtxt(args['sp_dir'] + f'{self.control_sp[i]}/xs.txt')
            us_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', us)
            ys_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', ys)
            us_all.append(np.tile(us, (self.N if i != 1 else self.N + self.Tini, 1)))
            ys_all.append(np.tile(ys, (self.N if i != 1 else self.N + self.Tini, 1)))
            us_all_.append(np.tile(us_, (self.N if i != 1 else self.N + self.Tini, 1)))
            ys_all_.append(np.tile(ys_, (self.N if i != 1 else self.N + self.Tini, 1)))
        self.us_all = np.concatenate(us_all, axis=0)
        self.ys_all = np.concatenate(ys_all, axis=0)
        self.us_all_ = np.concatenate(us_all_, axis=0)
        self.ys_all_ = np.concatenate(ys_all_, axis=0)
        datatool.savetxt(self.control_data_dir, us_all=self.us_all, ys_all=self.ys_all)
        
        # replace the set-point for deepc
        self.deepc.replace_sp(self.us_all, self.ys_all, self.us_all_, self.ys_all_, sp_num=len(self.control_sp))
        
        # load the plant
        if self.system == 'WWTPs':
            from .Plants.waste_water_system import waste_water_system as simulator
            x0, p = datatool.loadtxt(f'./_data/{system}/', 'ss_open', self.p_name) 
            self.plant = simulator(x0, p, x0_std_dev=self.x0_std_dev, us=us0, xs=xs0)
        elif self.system == 'threetanks':
            from .Plants.three_tanks import three_tank_system as simulator
            self.plant = simulator(x0_std_dev=self.x0_std_dev, us=us0, xs=xs0)
        elif self.system == 'siso':
            from .Plants.siso import siso as simulator
            self.plant = simulator(x0_std_dev=self.x0_std_dev, us=us0, xs=xs0)
        elif self.system == 'grn':
            from .Plants.grn import grn as simulator
            self.plant = simulator(xs=xs0)
        
        # load offline collected data
        self.offline_dir = args['offline_dir']
        # ud, yd, pd = datatool.loadtxt(self.offline_dir, 'u', 'y', 'p')
        ud, yd = datatool.loadtxt(self.offline_dir, 'u', 'y')
        
        ud_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', ud)   
        yd_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', yd)  
        
        # construct Hankel matrix,  past and future Hankel matrices
        Hlu_ = mathtool.hankel(self.Tini+self.Np, 'array', ud_[:self.T, :])
        Hly_ = mathtool.hankel(self.Tini+self.Np, 'array', yd_[:self.T, :])
        
        self.Up_, self.Uf_ = Hlu_[:self.u_dim*self.Tini, :].copy(), Hlu_[self.u_dim*self.Tini:, :].copy()
        self.Yp_, self.Yf_ = Hly_[:self.y_dim*self.Tini, :].copy(), Hly_[self.y_dim*self.Tini:, :].copy()
        
        # constraints 
        u_lb = np.array(args['u_lb']).reshape(1, -1)
        u_ub = np.array(args['u_ub']).reshape(1, -1)
        y_lb = np.array(args['y_lb']).reshape(1, -1)
        y_ub = np.array(args['y_ub']).reshape(1, -1)
        u_lb_, u_ub_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', u_lb, u_ub)
        y_lb_, y_ub_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', y_lb, y_ub)
        self.u_lb_, self.u_ub_, self.y_lb_, self.y_ub_ = u_lb_.reshape(-1, 1), u_ub_.reshape(-1, 1), y_lb_.reshape(-1, 1), y_ub_.reshape(-1, 1)
        self.u_lb_Np_ = np.tile(self.u_lb_, (self.Np, 1))
        self.u_ub_Np_ = np.tile(self.u_ub_, (self.Np, 1))
        self.y_lb_Np_ = np.tile(self.y_lb_, (self.Np, 1))
        self.y_ub_Np_ = np.tile(self.y_ub_, (self.Np, 1))
        
        # noise
        noise_dir = self.control_dir + 'noise.npy'
        x0_dir = self.control_dir + 'x0.npy'
        if os.path.exists(noise_dir):
            self.noise_data = np.load(noise_dir) if self.noise else None
        else:
            self.noise_data = None
        
        self.x0_data = np.load(x0_dir) if os.path.exists(x0_dir) else None
        return 
    
    
    def _init_con_opt(self):
        """ Initialize the constraint optimization solver """
        print("Constraint optimization solver initialization ...")
        
        ineqconidx = {'u': np.arange(self.u_dim), 'y': np.arange(self.y_dim)}
        ineqconbd = {'lbu': self.u_lb_[:, 0], 'ubu': self.u_ub_[:, 0], 'lby': self.y_lb_[:, 0], 'uby': self.y_ub_[:, 0]}     
           
        con_solver = con_opt(self.g_dim, self.Np, self.Uf_, self.Yf_, ineqconidx, ineqconbd)
        con_solver.init_solver(self.solver, self.dpc_opts)
        return con_solver    
    
    
    def _check_constraints(self, u_, y_):
        """ Check if the constraints are satisfied """
        ulb_check = np.all(u_ >= self.u_lb_Np_)
        uub_check = np.all(u_ <= self.u_ub_Np_)
        ylb_check = np.all(y_ >= self.y_lb_Np_)
        yub_check = np.all(y_ <= self.y_ub_Np_)
        check_flag = ulb_check and uub_check and ylb_check and yub_check
        return check_flag
    
     
    def _update_ini(self, uini_, yini_, uk_=None, yk_=None, us_=None, ys_=None):
        """
        Args:
            uk_ (_type_, optional): _description_. Defaults to None.
            yk_ (_type_, optional): _description_. Defaults to None.

        Returns:
            input: (1, (u_dim + y_dim)*Tini), input to the neural network
        """
        if uk_ is not None and yk_ is not None:
            uini_ = np.concatenate((uini_[1:, :].copy(), uk_.copy().reshape(1, -1)), axis=0)
            yini_ = np.concatenate((yini_[1:, :].copy(), yk_.copy().reshape(1, -1)), axis=0)
            
        uini_line, yini_line = uini_.copy().reshape(1, -1), yini_.copy().reshape(1, -1)
        us_, ys_ = us_.copy().reshape(1, -1), ys_.copy().reshape(1, -1)
        u_e_ = us_ - uini_[-1, :].copy()
        y_e_ = ys_ - yini_[-1, :].copy()
        inputs = torch.FloatTensor(np.hstack((uini_line, yini_line, u_e_, y_e_))).to(self.device)
        # inputs = torch.FloatTensor(np.hstack((uini_line, yini_line))).to(self.device)
        return inputs, uini_, yini_
    
    
    def _deepc(self, test_num=100, X0=None, Noise=None):
        """ Run the deep test
        Args:
            test_num (int, optional): The numbers of test with different x0. Defaults to 100.
            X0 (list, optional): The initial states of the plant. Defaults to None.
            Noise (list, optional): The noise of the plant. Defaults to None.
        """
        U_dpc, Y_dpc = [], []
        t_dpc = []
        for i in range(test_num):
            x0 = X0[i] if X0 is not None else None
            noise = Noise[i] if Noise is not None else None
            udpc, ydpc, _, _, _, _, t_ = self.deepc.generate(x0, noise)
            t_dpc.append(t_)
            U_dpc.append(udpc)
            Y_dpc.append(ydpc)
        U_dpc, Y_dpc = np.stack(U_dpc), np.stack(Y_dpc)
        U_dpc_mean, Y_dpc_mean = np.mean(U_dpc, axis=0), np.mean(Y_dpc, axis=0)
        U_dpc_var, Y_dpc_var= np.var(U_dpc, axis=0), np.var(Y_dpc, axis=0)
        t_dpc_mean = np.mean(t_dpc)
        return U_dpc_mean, Y_dpc_mean, U_dpc_var, Y_dpc_var, t_dpc_mean
    
    
    def _open_loop(self, test_num=100, X0=None, Noise=None):
        """ Run the open-loop test 
        Args:
            test_num (int, optional): The numbers of test with different x0. Defaults to 100.
            X0 (list, optional): The initial states of the plant. Defaults to None.
        """
        U_test, Y_test = [], []
        probar = progressbar.probar1()
        with probar:
            task = probar.add_task('Open-loop tests', total=test_num, unit='test')
            
            for i in range(test_num):
                # collect Tini steps pid control sequences
                _ = self.plant.reset()
                x0 = X0[i] if X0 is not None else None
                noise = Noise[i] if Noise is not None else None
                self.plant.set_initial(x0, noise)
                
                for i in range(self.Tini + self.N * len(self.control_sp)):
                    uk = self.us_all[i, :].copy()
                    
                    _ = self.plant.step(uk, noise=self.noise)
                    
                y_test = np.array(self.plant.state_buffer.memory)[:-1, self.y_idx]
                Y_test.append(y_test)
                probar.update(task, advance=1)
            Y_test = np.stack(Y_test)
            Y_mean, Y_var = np.mean(Y_test, axis=0), np.var(Y_test, axis=0)
        return Y_mean, Y_var

    
    def _yu_plot(self, Y_mean, Y_var, U_mean, U_var, Uini_mean, Uini_var, Yini_mean, Yini_var, \
                YLoss_mean=None, YLoss_var=None, Y_open_mean=None, Y_open_var=None, \
                U_dpc_mean=None, Y_dpc_mean=None, U_dpc_var=None, Y_dpc_var=None, test_num=None):
        
        if Y_open_mean is not None and Y_open_var is not None:
            flag_openloop = True
        else: 
            flag_openloop = False
        
        ypath = np.vstack((Yini_mean, Y_mean))
        upath = np.vstack((Uini_mean, U_mean))
        ypath_var = np.vstack((Yini_var, Y_var))
        upath_var = np.vstack((Uini_var, U_var))
        t = np.arange(self.N * len(self.control_sp) + self.Tini) * self.plant.sampling_period
        t_step, [upath_step, upath_var_step, us_step, ys_step, u_dpc_mean_step, u_dpc_var_step] \
            = datatool.data_to_step(t, upath, upath_var, self.us_all, self.ys_all, U_dpc_mean, U_dpc_var)
        
        datatool.savetxt(self.control_data_dir, ypath=ypath, upath=upath, ypath_var=ypath_var, upath_var=upath_var, t=t, t_step=t_step, \
            upath_step=upath_step, upath_var_step=upath_var_step, us_step=us_step, ys_step=ys_step)
        
        fig, ax = plt.subplots(ncols=3, nrows=2, sharex=True, figsize=(16, 8))

        tini = self.Tini * self.plant.sampling_period
        
        for i in range(3):
            ax[0, i].fill_between(t[t <= tini], ypath[t <= tini, i] - ypath_var[t <= tini, i], ypath[t <= tini, i] + ypath_var[t <= tini, i], color='lightskyblue', alpha=0.4, label='PID variance')
            ax[0, i].fill_between(t[t >= tini], ypath[t >= tini, i] - ypath_var[t >= tini, i], ypath[t >= tini, i] + ypath_var[t >= tini, i], color='tomato', alpha=0.4, label='DDeePC variance')
            ax[0, i].fill_between(t[t >= tini], Y_dpc_mean[t >= tini, i] - Y_dpc_var[t >= tini, i], Y_dpc_mean[t >= tini, i] + Y_dpc_var[t >= tini, i], color='greenyellow', alpha=0.4, label='DeePC variance')
            ax[0, i].fill_between(t, Y_open_mean[:, i] - Y_open_var[:, i], Y_open_mean[:, i] + Y_open_var[:, i], color='orange', alpha=0.4, label='Open-loop variance') if flag_openloop else None
            ax[0, i].plot(t, Y_open_mean[:, i], '-.', color='darkorange', label='Open-loop mean') if flag_openloop else None
            ax[0, i].plot(t[t <= tini], ypath[t <= tini, i], '-', color='darkviolet', label='PID mean')
            ax[0, i].plot(t[t >= tini], ypath[t >= tini, i], '-', color='red', label='DDeePC mean')
            ax[0, i].plot(t[t >= tini], Y_open_mean[t >= tini, i], '-.', color='dodgerblue', label='DeePC mean')
            ax[0, i].axvline(x=tini, linestyle='--', color='b', label='Tini')
            ax[0, i].plot(t_step, ys_step[:, i], linestyle='--', color='g', label='Set-point')
            ax[0, i].set_xlabel('Time (day)')
            ax[0, i].set_ylabel(f"y{i+1}")
            
        for i in range(3):
            mask_before = t_step <= tini
            mask_after = t_step >= tini
            
            t_step_before = t_step[mask_before]
            upath_step_before = upath_step[mask_before, i]
            u_dpc_mean_step_before = u_dpc_mean_step[mask_before, i]
            upath_var_step_before = upath_var_step[mask_before, i]
            u_dpc_mean_step_before = u_dpc_mean_step[mask_before, i]
            
            t_step_after = t_step[mask_after]
            upath_step_after = upath_step[mask_after, i]
            u_dpc_mean_step_after = u_dpc_mean_step[mask_after, i]
            upath_var_step_after = upath_var_step[mask_after, i]
            u_dpc_var_step_after = u_dpc_var_step[mask_after, i]
            
            
            ax[1, i].fill_between(t_step_before, upath_step_before - upath_var_step_before, upath_step_before + upath_var_step_before, color='lightskyblue', alpha=0.4, label='PID variance')
            ax[1, i].fill_between(t_step_after, upath_step_after - upath_var_step_after, upath_step_after + upath_var_step_after, color='orange', alpha=0.4, label='DDeePC variance')
            ax[1, i].fill_between(t_step_after, u_dpc_mean_step_after - u_dpc_var_step_after, u_dpc_mean_step_after + u_dpc_var_step_after, color='yellowgreen', alpha=0.4, label='DeePC variance')
            ax[1, i].plot(t_step_before, upath_step_before, '-', color='darkviolet', label='PID mean')
            ax[1, i].plot(t_step_after, upath_step_after, '-', color='red', label='DDeePC mean')
            ax[1, i].plot(t_step_after, u_dpc_mean_step_after, '-.', color='dodgerblue', label='DeePC mean')
            ax[1, i].axvline(x=tini, linestyle='--', color='b', label='Tini')
            ax[1, i].plot(t_step, us_step[:, i], linestyle='--', color='g', label='Set-point')
            ax[1, i].set_xlabel('Time (day)')
            ax[1, i].set_ylabel(f"u{i+1}")
        ax[0, 0].legend()
        fig.suptitle(f"average y loss: {np.mean(YLoss_mean)}\n x0 std dev: {self.x0_std_dev}\n test number: {test_num}")
        fig.savefig(self.fig_dir + f"/control_yu.pdf")
        plt.close()
        
        fig2 = plt.figure(figsize=(9, 6))
        plt.fill_between(t[t >= tini], YLoss_mean - YLoss_var, YLoss_mean + YLoss_var, color='orange', alpha=0.4, label='Variance')
        plt.plot(t[t >= tini], YLoss_mean, '-r', label='Mean')
        plt.xlabel('Time (day)')
        plt.ylabel('y loss')
        plt.legend()
        plt.suptitle(f"average y loss: {np.mean(YLoss_mean)}\n x0 std dev: {self.x0_std_dev}\n test number: {test_num}")
        fig2.savefig(self.fig_dir + f"/control_yloss.pdf")
        plt.close()
        
    
    @timer
    def rollout(self, test_num=100):
        """
            Run the DeePC control closed-loop test
        Args:
            test_num (int, optional): The numbers of test with different x0. Defaults to 100.
        """
        Utest, Utest_, Ytest, Ytest_, YLoss, Uini, Yini, X0, Noise, t_con, t_nn, event_trigger_list = [], [], [], [], [], [], [], [], [], [], [], []
        probar = progressbar.probar1()
        with probar:
            task = probar.add_task('Control tests', total=test_num, unit='test')
            for j in range(test_num):
                # collect Tini steps pid control sequences
                _ = self.plant.reset()
                noise = self.plant.generate_noise(self.Tini + self.N * len(self.control_sp)) if self.noise_data is None else self.noise_data[j, :, :]
                x0 = self.plant.state.copy() if self.x0_data is None else self.x0_data[j, :]
                self.plant.set_initial(x0, noise)
                X0.append(x0)
                Noise.append(noise)
                uini0, yini0 = [], []
                event_list = []
                
                # PID_bar = tqdm(total=self.Tini, desc=f"PID ini")
                for i in range(self.Tini):
                    uk = self.plant.get_action_pid()
                    
                    uini0.append(uk)
                    yini0.append(self.plant.state[self.y_idx].copy())
                    
                    _ = self.plant.step(uk, noise=self.noise)
                    
                    # PID_bar.update()
                uini0, yini0 = np.stack(uini0), np.stack(yini0)
                
                # Implement DeePC
                uini, yini = uini0.copy(), yini0.copy()
                uini_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', uini)
                yini_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', yini)
                
                U, U_, Y, Y_, yloss = [], [], [], [], []
                t_con_opt, t_nn_opt, num_con = [], [], []
                uk_, yk_, yk1_ = None, None, None
                
                # deepc_bar = tqdm(total=self.N, desc=f"DeePC")
                for i in range(self.N * len(self.control_sp)):
                    us_ = self.us_all_[i+self.Tini:i+self.Tini+1, :].copy()
                    # ys_ = self.ys_all_[i+self.Tini:i+self.Tini+1, :].copy()
                    if i + 2 < self.N * len(self.control_sp):
                        ys_ = self.ys_all_[i+self.Tini+2:i+self.Tini+3, :].copy()  # y^r_k+1
                    else:
                        ys_ = self.ys_all_[-1:, :].copy()
                    
                    inputs, uini_, yini_ = self._update_ini(uini_, yini_, uk_, yk1_, us_, ys_)

                    with torch.no_grad():
                        t_ = time.time()
                        output = self.model(inputs.detach())
                        t_nn_s = time.time() - t_
                    g = output.T.cpu().detach().numpy()
                    u_ = self.Uf_ @ g
                    y_ = self.Yf_ @ g
                    
                    # check if constraints are satisfied
                    if self.con_opt:
                        check_flag = self._check_constraints(u_, y_)
                        event_list.append(0 if check_flag else 1)  # 0 means no event, 1 means event triggered
                        if not check_flag:
                            g, t_s = self.con_solver.solver_step(g)
                            t_con_opt.append(t_s)
                            u_ = self.Uf_ @ g
                            # y_ = self.Yf_ @ g
                            # check_flag = self._check_constraints(u_, y_)
                            # print(check_flag)
                            t_nn_s += t_s
                    uk_ = u_[:self.u_dim, 0].copy()
                    uk = datatool.unscale(self.u_sc1, self.u_sc2, 'minmax', 'array', uk_)
                    
                    yk = self.plant.state[self.y_idx].copy()
                    yk_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', yk)
                    
                    _ = self.plant.step(uk.copy(), noise=self.noise)
                    
                    # here yk1_ should be yk, and yk should be yk_1
                    yk1 = self.plant.state[self.y_idx].copy()
                    yk1_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', yk1)
                    
                    step_loss = np.linalg.norm(yk_ - ys_, ord=2)
                    
                    U.append(uk)
                    U_.append(uk_)
                    Y.append(yk)
                    Y_.append(yk_)
                    yloss.append(step_loss)
                    t_nn_opt.append(t_nn_s)
                    
                    # deepc_bar.set_postfix(yloss=f"{step_loss}", ave_yloss=f"{np.mean(yloss)}")
                    # deepc_bar.update()
                
                U, U_, Y, Y_, yloss, ave_yloss = np.stack(U), np.stack(U_), np.stack(Y), np.stack(Y_), np.stack(yloss), np.mean(yloss)
                # progress_bar.update()
                
                Uini.append(uini0)
                Yini.append(yini0)
                Utest.append(U)
                Utest_.append(U_)
                Ytest.append(Y)
                Ytest_.append(Y_)
                YLoss.append(yloss)
                t_con.append(np.mean(t_con_opt) if len(t_con_opt) > 0 else 0)
                num_con.append(len(t_con_opt) if len(t_con_opt) > 0 else 0)
                event_trigger_list.append(event_list)
                t_nn.append(np.mean(t_nn_opt))
                probar.update(task, advance=1)
        
        Uini, Yini = np.stack(Uini), np.stack(Yini)
        Utest, Utest_, Ytest, Ytest_, YLoss = np.stack(Utest), np.stack(Utest_), np.stack(Ytest), np.stack(Ytest_), np.stack(YLoss)
        U_mean, U_var = np.mean(Utest, axis=0), np.var(Utest, axis=0)
        Y_mean, Y_var = np.mean(Ytest, axis=0), np.var(Ytest, axis=0)
        Uini_mean, Uini_var = np.mean(Uini, axis=0), np.var(Uini, axis=0)
        Yini_mean, Yini_var = np.mean(Yini, axis=0), np.var(Yini, axis=0)
        YLoss_mean, YLoss_var = np.mean(YLoss, axis=0), np.var(YLoss, axis=0)
        t_con_mean = np.array([np.mean(t_con)])
        t_nn_mean = np.array([np.mean(t_nn)])
        num_con_opt_mean = np.array(sum(num_con) / test_num) if self.con_opt else np.array([0])
        event_trigger_list = np.array(event_trigger_list)

        # Run deepc test for comparison
        U_dpc_mean, Y_dpc_mean, U_dpc_var, Y_dpc_var, t_dpc_mean = self._deepc(test_num, X0, Noise)
        
        # # Run open-loop test for comparison
        Y_open_mean, Y_open_var = self._open_loop(test_num, X0, Noise) if self.open_loop else (None, None)
        
        
        datatool.savetxt(self.control_data_dir, u_mean=U_mean, u_var=U_var, y_mean=Y_mean, y_var=Y_var, yloss_mean=YLoss_mean,\
            yloss_var=YLoss_var, Y_open_mean=Y_open_mean, Y_open_var=Y_open_var, U_dpc_mean=U_dpc_mean, Y_dpc_mean=Y_dpc_mean, \
            U_dpc_var=U_dpc_var, Y_dpc_var=Y_dpc_var, t_con_opt=t_con_mean, t_nn_mean=t_nn_mean, t_dpc_mean=t_dpc_mean, \
            num_con_mean=num_con_opt_mean, event_trigger_list=event_trigger_list, x0=np.array(X0))
        np.save(self.control_dir + 'noise.npy', np.array(Noise))  # 3D array, [test_num, N, noise_dim]
        np.save(self.control_dir + 'x0.npy', np.array(X0))  # 2D array, [test_num, x0_dim]
        self._yu_plot(Y_mean, Y_var, U_mean, U_var, Uini_mean, Uini_var, Yini_mean, Yini_var, YLoss_mean, YLoss_var, Y_open_mean, Y_open_var, U_dpc_mean, Y_dpc_mean, U_dpc_var, Y_dpc_var, test_num)      
    
    
   
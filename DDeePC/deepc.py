"""
Name: deepc.py
Author: Xuewen Zhang
Date: 18/06/2024
Project: DeepDeePC_offline copied to DDeePC 
Description: Generate trajectories for comparison of the DeePC trajectories,
             original name 'data_generation.py'
"""


import numpy as np
import matplotlib.pyplot as plt

from QYtool import *
import deepctools as dpctools



class deepc():
    
    def __init__(self, **kwargs):
        # file directory
        self.fig_dir = kwargs.get('fig_dir', '_data/')
        self.data_dir = kwargs.get('data_dir', '_data/')
        # system paramters
        self.system = kwargs.get('system', 'threetanks')
        self.noise = kwargs.get('noise', False)
        self.u_dim = kwargs.get('u_dim', 3)
        self.y_dim = kwargs.get('y_dim', 3)
        self.x0_std_dev = kwargs.get('x0_std_dev', 0.03)
        
        
        # load the plant simulator
        if self.system == 'threetanks':
            from .Plants.three_tanks import three_tank_system as simulator
            self.plant = simulator(x0_std_dev=self.x0_std_dev)
        elif self.system == 'WWTPs':
            from .Plants.waste_water_system import waste_water_system as simulator
            p_name = kwargs.get('p_name', 'inf_rain_mean')
            x0, pdata = datatool.loadtxt(f'./_data/{system}/', 'ss_open', p_name) 
            self.plant = simulator(x0, pdata, x0_std_dev=self.x0_std_dev)
        elif self.system == 'bilinear_motor':
            from .Plants.bilinear_motor import bilinear_motor as simulator
            self.plant = simulator(x0_std_dev=self.x0_std_dev)
        elif self.system == 'siso':
            from .Plants.siso import siso as simulator
            self.plant = simulator(x0_std_dev=self.x0_std_dev)
        elif self.system == 'grn':
            from .Plants.grn import grn as simulator
            self.plant = simulator()
        self.y_idx = self.plant.observed_dims
        
        # deepc parameters
        self.N = kwargs.get('N', 200)   # the steps for one set-point
        self.T = kwargs.get('T', 100)
        self.Tini = kwargs.get('Tini', 10)
        self.Np = kwargs.get('Np', 10)
        self.Q = np.diag(np.tile(kwargs.get('Q', np.ones(self.y_dim)), self.Np))
        self.R = np.diag(np.tile(kwargs.get('R', np.ones(self.u_dim)), self.Np))
        self.lambda_y = np.diag(np.tile(kwargs.get('lambda_y', 100*np.ones(self.y_dim)), self.Tini))
        self.lambda_g = kwargs.get('lambda_g', 10) * np.eye(self.T-self.Tini-self.Np+1)
        self.RDeePC = kwargs.get('RDeePC', False)
        self.dpc_opts = kwargs.get('dpc_opts', {})
        self.sp_dir = kwargs.get('sp_dir', '_data/threetanks/setpoints/')
        self.sp_num = kwargs.get('sp_num', 100)
        self.svd = kwargs.get('svd', False)
        self.uloss = kwargs.get('uloss', 'uus')

        # data process
        self._data_process(**kwargs)
        
        # init DeePC
        self._init_dpc()
        return 
        
        
    def _data_process(self, **kwargs):
        """ Data processing """
        # offline collected data for DeePC
        offline_dir = kwargs.get('offline_dir', '_data/threetanks/offline_data/')
        ud, yd = datatool.loadtxt(offline_dir, 'u', 'y')
        ud = ud.reshape(-1, self.u_dim)
        yd = yd.reshape(-1, self.y_dim)
        
        # data for scale
        scale_dir = kwargs.get('scale_dir', '_data/threetanks/scale_data/minmax/')
        u_sc1, u_sc2 = datatool.loadtxt(scale_dir, 'u_min', 'u_max')
        y_sc1, y_sc2 = datatool.loadtxt(scale_dir, 'y_min', 'y_max')
        # p_sc1, p_sc2 = datatool.loadtxt(self.scale_dir, 'p_min', 'p_max')
        
        self.u_sc1, self.u_sc2, self.y_sc1, self.y_sc2 = \
            u_sc1.reshape(1, -1), u_sc2.reshape(1, -1), y_sc1.reshape(1, -1), y_sc2.reshape(1, -1)
        
        ud_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', ud)   
        yd_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', yd)     
        # pd_, p_ = datatool.scale(self.p_sc1, self.p_sc2, 'minmax', 'array', pd, p)
        self.ud_, self.yd_ = ud_[:self.T, :], yd_[:self.T, :]

        # set-points trajectories of deepc
        self.us_all, self.ys_all, self.us_all_, self.ys_all_ = self._sp_process()
        
        # bounds of the control input and controlled output
        u_lb = np.array(kwargs.get('u_lb', self.plant.action_low.copy())).reshape(1, -1)
        u_ub = np.array(kwargs.get('u_ub', self.plant.action_high.copy())).reshape(1, -1)
        y_lb = np.array(kwargs.get('y_lb', self.plant.y_low.copy())).reshape(1, -1)
        y_ub = np.array(kwargs.get('y_ub', self.plant.y_high.copy())).reshape(1, -1)
        
        self.u_lb_, self.u_ub_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', u_lb, u_ub)
        self.y_lb_, self.y_ub_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', y_lb, y_ub)

        self.u_lb_Np_ = np.tile(self.u_lb_, (self.Np, 1)).reshape(-1, 1)
        self.u_ub_Np_ = np.tile(self.u_ub_, (self.Np, 1)).reshape(-1, 1)
        self.y_lb_Np_ = np.tile(self.y_lb_, (self.Np, 1)).reshape(-1, 1)
        self.y_ub_Np_ = np.tile(self.y_ub_, (self.Np, 1)).reshape(-1, 1)
        return 


    def _sp_process(self):
        """ Process the set-points trajectories of deepc """
        us_all_, ys_all_ = [], []
        us_all, ys_all = [], []
        sp_idx = np.arange(1, self.sp_num+1)
        # shuffle the set-points
        np.random.shuffle(sp_idx)
        j = 0
        for i in sp_idx:
            j += 1
            us, ys = datatool.loadtxt(self.sp_dir+f'{i}/', 'us', 'ys')
            us_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', us)
            ys_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', ys)
            us_all.append(np.tile(us, (self.N if j != 1 else self.N + self.Tini, 1)))
            ys_all.append(np.tile(ys, (self.N if j != 1 else self.N + self.Tini, 1)))
            us_all_.append(np.tile(us_, (self.N if j != 1 else self.N + self.Tini, 1)))
            ys_all_.append(np.tile(ys_, (self.N if j != 1 else self.N + self.Tini, 1)))
        us_all = np.concatenate(us_all, axis=0)
        ys_all = np.concatenate(ys_all, axis=0)
        us_all_ = np.concatenate(us_all_, axis=0)
        ys_all_ = np.concatenate(ys_all_, axis=0)
        return us_all, ys_all, us_all_, ys_all_
    
    
    def replace_sp(self, us_all, ys_all, us_all_, ys_all_, sp_num):
        """ replace the tracking set-points with the new set-points """
        self.us_all, self.ys_all, self.us_all_, self.ys_all_ = us_all, ys_all, us_all_, ys_all_
        self.sp_num = sp_num
        return 


    def _init_dpc(self):
        """
            initialize DeePCtool
        """
        self.u_real_idx = [i for i in range(self.u_dim)]   # except DeepAugKoopman, others first u_dim are real u

        # init deepc tool
        dpc_args = [self.u_dim, self.y_dim, self.T, self.Tini, self.Np, self.ud_, self.yd_, self.Q, self.R]
        dpc_kwargs = dict(
                          sp_change=True,
                          lambda_g=self.lambda_g,
                          lambda_y=self.lambda_y,
                          ineqconidx={'u': self.u_real_idx, 'y': np.arange(self.y_dim)},
                          ineqconbd={'lbu': self.u_lb_.tolist(), 'ubu': self.u_ub_.tolist(), 'lby': self.y_lb_.tolist(), 'uby': self.y_ub_.tolist()},
                          svd=self.svd  # TODO: added svd
                          )
        self.dpc = dpctools.deepctools(*dpc_args, **dpc_kwargs)

        # init and formulate deepc solver
        # dpc_opts = {
        #     'ipopt.max_iter': 1000,  # 50
        #     'ipopt.tol': 1e-5,
        #     'ipopt.print_level': 1,
        #     'print_time': 0,
        #     # 'ipopt.acceptable_tol': 1e-8,
        #     # 'ipopt.acceptable_obj_change_tol': 1e-6,
        # }
        if self.RDeePC:  # if true: Robust DeePC
            self.dpc.init_RDeePCsolver(uloss=self.uloss, opts=self.dpc_opts)
        else:
            self.dpc.init_DeePCsolver(uloss=self.uloss, opts=self.dpc_opts)
        return 
        
    
    def _update_dpc(self, step, uini, yini, uk=None, yk=None):
        """ Update one step of DeePC
        Args:
            step (int): the current step
            uini (np.ndarray)(u_dim*Tini, 1): the initial control input
            yini (np.ndarray)(y_dim*Tini, 1): the initial controlled output
            uk (np.ndarray)(u_dim, 1): the current state
            yk (np.ndarray)(y_dim, 1): the current controlled output
        """
        if uk is not None and yk is not None:
            uini = np.concatenate([uini[self.u_dim:, :], uk], axis=0)
            yini = np.concatenate([yini[self.y_dim:, :], yk], axis=0)
        uref = np.tile(self.us_all_[step, :].copy(), self.Np).reshape(-1, 1)
        yref = np.tile(self.ys_all_[step, :].copy(), self.Np).reshape(-1, 1)
        u_opt, g_opt, t_s = self.dpc.solver_step(uini, yini, uref, yref)
        return u_opt, g_opt, t_s, uini, yini
    
    
    def _pid_loop(self):
        """ initialize the deepc with PID controller """
        pid_bar = progressbar.probar2()
        # apply pid control if x0 != xs_0, the first set-point, otherwise, use us_0
        pid_flag = True if self.plant.state[self.y_idx].all() != self.ys_all[0,:].all() else False
        with pid_bar:
            pid_task = pid_bar.add_task("PID loop", unit='step', total=self.Tini, loss=np.inf)
            cost_y_ini = []
            for i in range(self.Tini):
                # plant simulation
                uk = self.plant.get_action_pid() if pid_flag else self.us_all[0, :]
                xk1 = self.plant.step(uk, self.noise)[0]
                
                yk_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', self.plant.state_buffer.memory[-2][self.y_idx].reshape(1, -1))
                cost_y_ = np.linalg.norm(yk_ - self.ys_all_[i:i+1, :])
                cost_y_ini.append(cost_y_)
                
                pid_bar.update(pid_task, advance=1, loss=cost_y_)
            
        uini = np.stack(self.plant.state_buffer.memory_action) if self.system != 'siso' else np.stack(self.plant.state_buffer.memory_action[:-1])
        yini = np.stack(self.plant.state_buffer.memory[:-1])[:, self.y_idx]
        uini_ = datatool.scale(self.u_sc1, self.u_sc2, 'minmax', 'array', uini)
        yini_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', yini)
        cost_y_ini = np.array(cost_y_ini)
        return uini, yini, uini_, yini_, cost_y_ini
    
    
    def _deepc_loop(self, uini, yini, sp_num=2):
        """ DeePC control loop 
        Args:
            uini (np.ndarray)(u_dim*Tini, 1): the scaled initial control input from PID controller
            yini (np.ndarray)(y_dim*Tini, 1): the scaled initial controlled output from PID controller
            sp_num (int): the number of set-points to be tested
        """
        dpc_bar = progressbar.probar2()
        with dpc_bar:
            dpc_task = dpc_bar.add_task("DeePC loop", unit='step', total=sp_num*self.N, loss=np.inf)
            uk_, yk_, t_memory, cost_y_dpc = None, None, [], []
            G, Uini, Yini, Eu, Ey = [], [], [], [], []
            for i in range(0, self.N*sp_num):
                step = i+self.Tini
                u_opt, g_opt, t_s, uini, yini = self._update_dpc(step, uini, yini, uk_, yk_)
                t_memory.append(t_s)
                
                # save data for deep training
                if i != 0:   # first step, uk_, yk_ are None, so skip
                    G.append(g_opt)    # TODO: check g_opt size, change to (g_dim,)
                    Uini.append(uini[:, 0])
                    Yini.append(yini[:, 0])
                    Eu.append(self.us_all_[step, :] - uk_[:, 0])
                    Ey.append(self.ys_all_[step, :] - yk_[:, 0])
                
                uk_ = u_opt[:self.u_dim].copy().reshape(-1, 1)   # (u_dim, 1)
                uk = datatool.unscale(self.u_sc1, self.u_sc2, 'minmax', 'array', uk_.reshape(1, -1))[0]  # (u_dim,)
                yk = self.plant.state.copy()[self.y_idx]  # (y_dim,)
                yk_ = datatool.scale(self.y_sc1, self.y_sc2, 'minmax', 'array', yk.reshape(1, -1)).reshape(-1, 1)  # (y_dim, 1)
                cost_y = np.linalg.norm(yk_ - self.ys_all_[step, :].reshape(-1, 1))
                cost_y_dpc.append(cost_y)
                
                # plant simulation
                xk1 = self.plant.step(uk, self.noise)[0]
                
                dpc_bar.update(dpc_task, advance=1, loss=cost_y)
        Uini, Yini = np.stack(Uini), np.stack(Yini)
        G, Eu, Ey = np.stack(G), np.stack(Eu), np.stack(Ey)
        cost_y_dpc, t_memory = np.array(cost_y_dpc), np.array(t_memory)
        udpc = np.stack(self.plant.state_buffer.memory_action[self.Tini:])
        ydpc = np.stack(self.plant.state_buffer.memory[self.Tini:-1])[:, self.y_idx]
        return Uini, Yini, G, Eu, Ey, udpc, ydpc, cost_y_dpc, t_memory        

    
    def open_loop(self, x0=None, sp_num=2):
        """ Open loop of the process with the same control reference """
        self.plant.reset()
        self.plant.set_initial(x0)
        open_bar = progressbar.probar1()
        with open_bar:
            open_task = open_bar.add_task("Open-loop", unit='step', total=self.N*sp_num+self.Tini)
            for i in range(self.N*sp_num+self.Tini):
                uk = self.us_all[i, :].copy()
                self.plant.step(uk, self.noise)
                open_bar.update(open_task, advance=1)
        uopen = np.stack(self.plant.state_buffer.memory_action)
        yopen = np.stack(self.plant.state_buffer.memory[:-1])[:, self.y_idx]
        return uopen, yopen
    
    
    def test(self):
        """ Test the current config of DeePC can achieve good tracking performance 
            if yes, then generate training data for the offline training
            if no, then adjust the config of DeepC
        Args:
            uini0 (np.ndarray)(Tini, u_dim): trajectory of control input from PID controller
            yini0 (np.ndarray)(Tini, y_dim): trajectory of controlled output from PID controller
            udpc (np.ndarray)(sp_num*N, u_dim): trajectory of control input from DeePC
            ydpc (np.ndarray)(sp_num*N, y_dim): trajectory of controlled output from DeePC
            upath (np.ndarray)(sp_num*N+Tini, u_dim): Entire trajectory of control input from PID and DeePC
            ypath (np.ndarray)(sp_num*N+Tini, y_dim): Entire trajectory of controlled output from PID and DeePC 
            uopen (np.ndarray)(sp_num*N+Tini, u_dim): Entire trajectory of control input from open-loop
            yopen (np.ndarray)(sp_num*N+Tini, y_dim): Entire trajectory of controlled output from open-loop
        """
        rprint("[green]Test DeePC config with 2 set-points ...")
        # DeePC test
        self.plant.reset()
        x0 = self.plant.state.copy()
        test_sp_num = 2
        uini0, yini0, uini0_, yini0_, cost_y_ini = self._pid_loop()
        Uini, Yini, G, Eu, Ey, udpc, ydpc, cost_y_dpc, t_memory = self._deepc_loop(uini0_.reshape(-1, 1), yini0_.reshape(-1, 1), test_sp_num)
        upath = np.stack(self.plant.state_buffer.memory_action)
        ypath = np.stack(self.plant.state_buffer.memory[:-1])[:, self.y_idx]
        # open-loop test
        uopen, yopen = self.open_loop(x0, test_sp_num)
        # plot the results
        self._plot(upath, ypath, uopen, yopen, np.mean(cost_y_dpc), test=True)
        # rmse
        rmse_deepc = np.sqrt(np.mean((self.ys_all[self.Tini:self.N*test_sp_num+self.Tini, :] - ydpc)**2))
        rmse_open = np.sqrt(np.mean((self.ys_all[self.Tini:self.N*test_sp_num+self.Tini, :] - yopen[self.Tini:, :])**2))
        rprint(f"RMSE of DeePC: [cyan]{rmse_deepc:.5f}[/cyan] \n RMSE of open-loop: [cyan]{rmse_open:.5f}[/cyan]")
        return 
        
        
    def generate(self, x0=None, noise=None, plot=False, open_loop=False):
        """ Generate the training data for the offline training 
        Args:
            x0 (np.ndarray)(x_dim, 1): the initial state of the plant
            uini0 (np.ndarray)(Tini, u_dim): trajectory of control input from PID controller
            yini0 (np.ndarray)(Tini, y_dim): trajectory of controlled output from PID controller
            udpc (np.ndarray)(sp_num*N, u_dim): trajectory of control input from DeePC
            ydpc (np.ndarray)(sp_num*N, y_dim): trajectory of controlled output from DeePC
            upath (np.ndarray)(sp_num*N+Tini, u_dim): Entire trajectory of control input from PID and DeePC
            ypath (np.ndarray)(sp_num*N+Tini, y_dim): Entire trajectory of controlled output from PID and DeePC 
            uopen (np.ndarray)(sp_num*N+Tini, u_dim): Entire trajectory of control input from open-loop
            yopen (np.ndarray)(sp_num*N+Tini, y_dim): Entire trajectory of controlled output from open-loop
        """
        rprint("[green]Generate DeePC data for offline training ...")
        # DeePC test
        self.plant.reset()
        x0 = self.plant.state.copy() if x0 is None else x0
        self.plant.set_initial(x0, noise)
        # pid loop
        uini0, yini0, uini0_, yini0_, cost_y_ini = self._pid_loop()
        # deep loop
        Uini, Yini, G, Eu, Ey, udpc, ydpc, cost_y_dpc, t_memory = self._deepc_loop(uini0_.reshape(-1, 1), yini0_.reshape(-1, 1), self.sp_num)
        
        upath = np.stack(self.plant.state_buffer.memory_action)
        ypath = np.stack(self.plant.state_buffer.memory[:-1])[:, self.y_idx]
        
        # save the training data
        datatool.savetxt(self.data_dir, Uini=Uini, Yini=Yini, G=G, Eu=Eu, Ey=Ey)
        # open-loop test
        uopen, yopen = self.open_loop(x0, self.sp_num) if open_loop else (None, None)
        # plot the results
        self._plot(upath, ypath, uopen, yopen, np.mean(cost_y_dpc)) if open_loop and plot else None
        
        t_mean = np.mean(t_memory)
        return upath, ypath, uini0, yini0, udpc, ydpc, t_mean
    
    
    def _plot(self, upath, ypath, uopen, yopen, yloss_mean=None, test=False):
        """ plot the results """
        ncols = len(self.y_idx)
        t = np.arange(upath.shape[0]) * self.plant.sampling_period
        tini = self.Tini * self.plant.sampling_period
        t_step, [uopen_step, upath_step, ys_step] = datatool.data_to_step(t, uopen, upath, self.ys_all[:uopen.shape[0], :])
        
        fig1, ax = plt.subplots(nrows=2, ncols=ncols, sharex=True, figsize=(18, 8))
        for i in range(ncols):
            ax[0, i].plot(t, ypath[:, i], color='red', label='DeePC')
            ax[0, i].plot(t, yopen[:, i], color='orange', linestyle='--', label='Open-loop')
            ax[0, i].plot(t_step, ys_step[:, i], color='blue', linestyle='--', label='Reference')
            ax[0, i].axvline(x=tini, color='g', linestyle=':', label='Tini')
            ax[0, i].set_ylabel(f'$y_{i+1}$')
            ax[0, i].set_xlabel(f"Time {'(day)' if self.system == 'WWTPs' else '(hour)'}")
        ax[0, 0].legend()
        
        for i in range(ncols):
            ax[1, i].plot(t_step, uopen_step[:, i], color='blue', linestyle='--', label='Reference')
            ax[1, i].plot(t_step, upath_step[:, i], color='red', label='DeePC')
            ax[1, i].axvline(x=tini, color='g', linestyle=':', label='Tini')
            ax[1, i].set_ylabel(f'$u_{i+1}$')
            ax[1, i].set_xlabel(f"Time {'(day)' if self.system == 'WWTPs' else '(hour)'}")
        
        fig1.suptitle(f"{self.system} - {'RDeePC' if self.RDeePC else 'DeePC'} - ave yloss: {yloss_mean}\n T: {self.T} Tini: {self.Tini} Np: {self.Np}")
        plt.tight_layout()
        name = 'yu_deepc_open.pdf' if not test else 'yu_deepc_open_test.pdf'
        fig1.savefig(self.fig_dir + name, bbox_inches='tight')
        # plt.show() if test else None
        return 
        
"""
Name: ddeepc_args.py
Author: Xuewen Zhang
Date: 19/04/2024
Project: DeepDeePC
"""
from DDeePC import MyTool
import os
import json

tool = MyTool()

update_variables = dict(
        # System properties
        system='threetanks',    # 'threetanks', 'WWTPs', 'bilinear_motor', 'siso'
        u_dim=3,
        y_dim=3,
        p_dim=0,
        x0_name=5,
        p_name='inf_rain_mean',  # not used in this system
        y_idx=[2, 5, 8],
        noise=False,     # if add noise to the system
        x0_std_dev=0.03,
        
        # DeePC parameters
        T=200,
        Tini=10,
        Np=10,
        N=200,   # control steps for one set-points
        sp_num=2,  # the number of set-points to test the DeePC configuration
        Q=[5, 5, 5],
        R=[1, 1, 1],
        P_y=[10, 10, 10],
        P_u=[10, 10, 10],
        RDeePC=False,                         # training RDeePC or DeePC 
        lambda_y=[10, 10, 10],            # RDeePC: weight of y mismatch
        lambda_g=10,                          # RDeePC: weight of operator g
        svd=False,
        solver='ipopt',
        dpc_opts={
            'ipopt.max_iter': 1000,  # 50
            'ipopt.tol': 1e-5,
            'ipopt.print_level': 1,
            'print_time': 0,
            # 'ipopt.acceptable_tol': 1e-8,
            # 'ipopt.acceptable_obj_change_tol': 1e-6,
        },
        
        # constraints
        con_opt=True,
        u_lb=[2.8e9, 0.9e9, 2.8e9],
        u_ub=[3e9, 1.2e9, 3e9],
        y_lb=[480, 472, 474],
        y_ub=[493.5, 486, 488],
        
        # Deep Learning parameters
        hidden_size_list=[150, 150, 150],
        lr=0.0001,
        epoch=1000,
        batch_size=200,
        data_size=10000,
        
        # Control set-points
        control_sp = [6, 7, 8], 
        open_loop = True,
)


class build_args(object):
    def __init__(self, **kwargs):
        """
            Initialize the variables and build args.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.exp, self.exp_dir = tool.create_new_folder_with_max_number(f'./_results/{self.system}/')
        self.data_dir = self.exp_dir + '/data/'
        self.model_dir = self.exp_dir + '/model/'
        self.fig_dir = self.exp_dir + '/fig/'
        self.control_dir = self.exp_dir + '/control/'
        self.control_fig_dir = self.exp_dir + '/control_fig/'
        tool.makedir(self.data_dir, self.model_dir, self.fig_dir, self.control_dir, self.control_fig_dir)
        
        tool.copy_file('./ddeepc_args.py', self.exp_dir)
        
        self.args = self.build_args()
        
            
    def build_args(self) -> dict:
        args = dict(
            # exp
            exp=self.exp,
            exp_dir=self.exp_dir,
            control_dir=self.control_dir,
            
            # System properties
            system=self.system,
            u_dim=self.u_dim,
            y_dim=self.y_dim,
            p_dim=self.p_dim,
            x0_name=f'./_data/{self.system}/set-point/{self.x0_name}/',
            p_name=self.p_name,
            sp_dir=f'./_data/{self.system}/setpoints/',
            sys_param_dir=f'./_data/{self.system}/',
            y_idx=self.y_idx,
            noise=self.noise,
            x0_std_dev=self.x0_std_dev,   
            
            # DeePC parameters
            T=self.T,
            Tini=self.Tini,
            Np=self.Np,
            N=self.N,
            sp_num=self.sp_num,
            Q=self.Q,
            R=self.R,
            P_y=self.P_y,
            P_u=self.P_u,
            RDeePC=self.RDeePC,
            lambda_y=self.lambda_y,
            lambda_g=self.lambda_g,
            svd=self.svd,
            solver=self.solver,
            dpc_opts=self.dpc_opts,
            
            # constraints
            con_opt=self.con_opt,
            u_lb=self.u_lb,
            u_ub=self.u_ub,
            y_lb=self.y_lb,
            y_ub=self.y_ub,
            
            # DeepLearning parameters
            lr=self.lr,
            input_size=(self.u_dim + self.y_dim)*(self.Tini+1),  # input [uini_, yini_, u_e_, y_e_]
            hidden_size_list=self.hidden_size_list,
            output_size=(self.T-self.Tini-self.Np + 1),
            epoch=self.epoch,
            batch_size=self.batch_size,
            data_size=self.data_size,
            model_dir=self.model_dir,
            data_dir=self.data_dir,
            fig_dir=self.fig_dir,
            control_fig_dir=self.control_fig_dir,
            offline_dir=f'./_data/{self.system}/offline_data/',  
            online_dir=f'./_data/{self.system}/online_data/',
            openloop_dir=f'./_data/{self.system}/offline_data/uhold/',    # TODO: change the dataset
            scale_dir=f'./_data/{self.system}/scale_data/minmax/',
            
            # Control set-points
            control_sp=self.control_sp,
            open_loop=self.open_loop,
        )
        tool.save_config(args, self.exp_dir+'/', 'args')
        return args
        
if __name__ == '__main__':
    args = build_args(**update_variables)
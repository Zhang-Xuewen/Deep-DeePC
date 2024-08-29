import numpy as np
import time
import os
import re
import json
import shutil
import torch
import casadi as cs
import matplotlib.pyplot as plt
from threading import Thread
from pathlib import Path

plt.rcParams['figure.max_open_warning'] = 50

def timer(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        print('Time elapsed: {}'.format(time.time() - start))
        return ret
    return wrapper

class MyTool():

    def __init__(self):
        self.name = 'Xuewen'

    def create_new_folder_with_max_number(self, directory, prefix=''):
        """
            In the given directory, create a new folder that have a larger number than folders already exist.
            For example: In the directory, have folders mx1, mx2, then create folder mx3
            directory: given path
            prefix: the beginning part of the filenames that you want to check for.
        """
        # check if exit the path, if not, create it.
        self.makedir(directory)

        # Get a list of all folders in the directory
        folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

        # Filter out folders that match the specified pattern
        pattern = re.compile(rf"{prefix}(\d+)")
        matching_folders = [folder for folder in folders if pattern.match(folder)]

        # Extract numbers from matching folder names
        numbers = [int(pattern.match(folder).group(1)) for folder in matching_folders]

        # Find the maximum number
        max_number = max(numbers) if numbers else 0

        # Create a new folder with the maximum number incremented by 1
        new_folder_name = f"{prefix}{max_number + 1}"
        new_folder_path = os.path.join(directory, new_folder_name)
        os.makedirs(new_folder_path)

        return new_folder_name, new_folder_path

    def makedir(self, *args):
        """Create the directory"""
        for item in args:
            if not os.path.exists(item):
                os.makedirs(item)

    def current_dir(self):
        """Get the path of current file"""
        current_directory = os.path.dirname(os.path.realpath(__file__))
        return current_directory

    def upper_dir(self):
        """Get the path of upper file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        return upper_dir   
    
    def copy_file(self, source_file, destination_directory):
        """Copy the source_file to the destination directory, and generate an info.txt give the original copied file path"""
        try:
            self.makedir(destination_directory)

            # Copy the file to the destination directory
            copied_file_name = os.path.basename(source_file)
            destination_file = os.path.join(destination_directory, copied_file_name)
            shutil.copy(source_file, destination_file)

            # Generate a text file with the content of the original file path and copied file name
            original_file_name = os.path.splitext(copied_file_name)[0]  # Extract filename without extension
            txt_file_path = os.path.join(destination_directory, f"{original_file_name}_info.txt")
            with open(txt_file_path, "w") as txt_file:
                txt_file.write(f"Original file path: {source_file}\n")
                txt_file.write(f"Copied file name: {copied_file_name}")

            print(f">>> File '{source_file}' copied to '{destination_directory}' as '{copied_file_name}' with '{original_file_name}_info.txt' successfully.")
        except FileNotFoundError:
            raise ValueError(f"Source file {source_file} not found.")
        except PermissionError:
            raise ValueError("Permission denied.")
        except Exception as e:
            raise ValueError(f"An error occurred: {e}")

    def savetxt(self, savedir, **kwargs):
        for name, item in kwargs.items():
            if isinstance(item, torch.Tensor):
                [item] = self.toArray(item.cpu())
            np.savetxt(savedir + '%s.txt' % name, item)


    def loadtxt(self, loaddir, *args):
        data = []
        for name in args:
            data.append(np.loadtxt(loaddir + '%s.txt' % name))
        return data


    def loadpt(self, loaddir, device, *args):
        """
            Load pt files using torch
            loaddir: save path
            device: cpu or gpu
            *args: file name in string
        """
        data = []
        for name in args:
            data.append(torch.load(loaddir + '%s.pt' % name, map_location=device))
        return data
    
    
    def load_model(self, model, root_path, model_name='val_model'):
        """
        Load the model weights.
        Args:
            root_path: Path where the model .pt and .json file located.
            model_name: Name of the .pt
        Returns: model with well-trained parameters
        """
        try:
            model_path = Path(root_path, "{}.pt".format(model_name))
            # model_config_path = Path(root_path, "{}_config.json".format(model_name))
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path)
            else:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['nn_model'])
            print(">> Model weights loaded from {} successfully!\n".format(model_name))
            return model
        except:
            print(">> Fail to load model!\n")


    def load_config(self, savedir='', *args):
        """args: the str of the load josn file name"""
        data_list = []
        for name in args:
            config = open(savedir + '%s.json' % name)
            data = json.load(config)
            data_list.append(data)
        return data_list


    def save_datatoconfig(self, save_dir='', **kwargs):
        """Write the dict into txt"""
        for name, data in kwargs.items():
            file = open(save_dir + '%s.json' % name, 'w')
            file.write('{\n')

            count = 0

            for key, value in data.items():
                count += 1
                if count != len(data.keys()):
                    file.write(f'"{key}": {value},\n')
                else:
                    file.write(f'"{key}": {value}\n')

            file.write('}\n')
            file.close()


    def save_config(self, args, savedir, name):
        """
        Used for saving the configs (args) of the model.
        """
        json_dir_val = os.path.join(savedir, "{}.json".format(name))
        with open(json_dir_val, 'w') as fp:
            json.dump(args, fp, indent=4, default=self.toList)

    def add_line_to_file(self, file_path, line):
        """Add one line sentence to certain txt"""
        with open(file_path, 'a') as f:
            f.write(line + '\n')


    # data process
    def toArray(self, *args):
        """Turn data to array"""
        data_list = []
        for data in args:
            if isinstance(data, np.ndarray):
                data_list.append(data)
            else:
                data_list.append(np.array(data))
        return data_list


    def toList(self, obj):
        """Convert NumPy arrays to Python lists"""
        if not isinstance(obj, list):
            return obj.tolist()
        return obj


    def toTorch(self, *args):
        """Turn data to array"""
        data_list = []
        for data in args:
            if isinstance(data, torch.Tensor):
                data_list.append(data)
            else:
                data_list.append(torch.FloatTensor(data))
        return data_list


    def toGPU(self, *args):
        data_all = []
        for data in args:
            if torch.is_tensor(data):
                data = data.to(torch.device("cuda:0"))
            else:
                raise ValueError("Not torch")
            data_all.append(data)
        return data_all


    def scale(self, x, x_max, x_min):
        x_scale = (x - x_min)/(x_max - x_min)
        # check if have x_max - x_min == 0
        if len(x_max.shape) == 1:
            if torch.is_tensor(x_max):
                zero_idx = torch.where(x_max - x_min == 0)[0]
            else:
                zero_idx = np.where(x_max - x_min == 0)[0]
            if len(zero_idx) != 0:
                x_scale[zero_idx] = 0
        else:
            if torch.is_tensor(x_max):
                zero_idx = torch.where(x_max - x_min == 0)[0]
            else:
                zero_idx = np.where(x_max - x_min == 0)[0]
            if len(zero_idx) != 0:
                x_scale[:, zero_idx] = 0
        return x_scale


    def scale_all(self, x_max, x_min, *args):
        """Scale the data with same max and min"""
        data_scale = []
        for data in args:
            data_scale.append(self.scale(data, x_max, x_min))
        return data_scale


    def normscale(self, x, x_mean, x_std):
        """Scale the data with to standard normal distribution"""
        x_scale = (x - x_mean)/x_std
        return x_scale


    def normscale_all(self, x_mean, x_std, *args):
        """Scale the data with to standard normal distribution"""
        data_scale = []
        for data in args:
            data_scale.append(self.normscale(data, x_mean, x_std))
        return data_scale


    def unscale(self, x, x_max, x_min):
        x = x * (x_max - x_min) + x_min
        return x


    def unscale_all(self, x_max, x_min, *args):
        """Unscale the data with same max and min"""
        data_unscale = []
        for data in args:
            data_unscale.append(self.unscale(data, x_max, x_min))
        return data_unscale


    def unnormscale(self, x, x_mean, x_std):
        """Unscale the data from standard normal distributio"""
        x = x * x_std + x_mean
        return x


    def unnormscale_all(self, x_mean, x_std, *args):
        """Unscale the data from standard normal distributio"""
        data_unscale = []
        for data in args:
            data_unscale.append(self.unnormscale(data, x_mean, x_std))
        return data_unscale

    
    def data_to_step(self, x, t=None):
        """
            Transform the sequential data to step data
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
    
    
    def hankel(self, x, L, datatype='array'):
        """
            ------Construct Hankel matrix------
            x: data sequence (data_size, x_dim)
            L: row dimension of the hankel matrix
            T: data samples of data x
            return: H(x): hankel matrix of x  H(x): (x_dim*L, T-L+1)
                    H(x) = [x(0)   x(1) ... x(T-L)
                            x(1)   x(2) ... x(T-L+1)
                            .       .   .     .
                            .       .     .   .
                            .       .       . .
                            x(L-1) x(L) ... x(T-1)]
                    Hankel matrix of order L has size:  (x_dim*L, T-L+1)
        """
        if datatype == 'array':
            if not isinstance(x, np.ndarray):
                x = np.array(x)

            T, x_dim = x.shape

            Hx = np.zeros((L * x_dim, T - L + 1))
            for i in range(L):
                Hx[i * x_dim:(i + 1) * x_dim, :] = x[i:i + T - L + 1, :].T  # x need transpose to fit the hankel dimension
        
        elif datatype == 'tensor':
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)

            T, x_dim = x.shape

            Hx = torch.zeros((L * x_dim, T - L + 1))

            for i in range(L):
                Hx[i * x_dim:(i + 1) * x_dim, :] = x[i:i + T - L + 1, :].T
        else:
            raise ValueError(f'datatype must be "array" or "tensor", {datatype} not exists!')
        return Hx
    

    def MSE(self, data1, data2):
        if data1.shape != data2.shape:
            raise ValueError(f'data1 shape {data1.shape} != data2 shape {data2.shape}')
        else:
            mse = np.mean(np.square(data1 - data2))
        return mse


    def cal_mse(self, Ys, Y_mpc, Y_empc, Y_ol_true, Y_ol_fp, **kwargs):
        """step: divide the transient period and steady state period
            kwargs: give range [0:400] and its name
        """
        mse_dict = {}
        # mpc
        mse_dict['mse_mpc_all'] = self.MSE(Y_mpc, Ys)
        mse_dict['mse_mpc_all_y1'] = self.MSE(Y_mpc[:, 0], Ys[:, 0])
        mse_dict['mse_mpc_all_y2'] = self.MSE(Y_mpc[:, 1], Ys[:, 1])
        # empc
        mse_dict['mse_empc_all'] = self.MSE(Y_empc, Ys)
        mse_dict['mse_empc_all_y1'] = self.MSE(Y_empc[:, 0], Ys[:, 0])
        mse_dict['mse_empc_all_y2'] = self.MSE(Y_empc[:, 1], Ys[:, 1])
        # openloop_true
        # Y_ol_true = Y_ol[:N+1, :]
        mse_dict['mse_ol_true_all'] = self.MSE(Y_ol_true, Ys)
        mse_dict['mse_ol_true_all_y1'] = self.MSE(Y_ol_true[:, 0], Ys[:, 0])
        mse_dict['mse_ol_true_all_y2'] = self.MSE(Y_ol_true[:, 1], Ys[:, 1])
        # openloop_fp
        # Y_ol_fp = Y_ol[:N+1, :]
        mse_dict['mse_ol_fp_all'] = self.MSE(Y_ol_fp, Ys)
        mse_dict['mse_ol_fp_all_y1'] = self.MSE(Y_ol_fp[:, 0], Ys[:, 0])
        mse_dict['mse_ol_fp_all_y2'] = self.MSE(Y_ol_fp[:, 1], Ys[:, 1])

        for name, value in kwargs.items():
        # first 300 epochs--> transient period
            mse_dict['mse_mpc_%s'%name] = self.MSE(Y_mpc[value[0]:value[1], :], Ys[value[0]:value[1], :])
            mse_dict['mse_mpc_%s_y1'%name] = self.MSE(Y_mpc[value[0]:value[1], 0], Ys[value[0]:value[1], 0])
            mse_dict['mse_mpc_%s_y2'%name] = self.MSE(Y_mpc[value[0]:value[1], 1], Ys[value[0]:value[1], 1])

            mse_dict['mse_empc_%s' % name] = self.MSE(Y_empc[value[0]:value[1], :], Ys[value[0]:value[1], :])
            mse_dict['mse_empc_%s_y1' % name] = self.MSE(Y_empc[value[0]:value[1], 0], Ys[value[0]:value[1], 0])
            mse_dict['mse_empc_%s_y2' % name] = self.MSE(Y_empc[value[0]:value[1], 1], Ys[value[0]:value[1], 1])

            # first 300 epochs--> transient period
            mse_dict['mse_ol_true_%s'%name] = self.MSE(Y_ol_true[value[0]:value[1], :], Ys[value[0]:value[1], :])
            mse_dict['mse_ol_true_%s_y1'%name] = self.MSE(Y_ol_true[value[0]:value[1], 0], Ys[value[0]:value[1], 0])
            mse_dict['mse_ol_true_%s_y2'%name] = self.MSE(Y_ol_true[value[0]:value[1], 1], Ys[value[0]:value[1], 1])
            # first 300 epochs--> transient period
            mse_dict['mse_ol_fp_%s'%name] = self.MSE(Y_ol_fp[value[0]:value[1], :], Ys[value[0]:value[1], :])
            mse_dict['mse_ol_fp_%s_y1'%name] = self.MSE(Y_ol_fp[value[0]:value[1], 0], Ys[value[0]:value[1], 0])
            mse_dict['mse_ol_fp_%s_y2'%name] = self.MSE(Y_ol_fp[value[0]:value[1], 1], Ys[value[0]:value[1], 1])
        return mse_dict


    def save_model(self, model, savedir, train_loss, train_loss_all, val_loss=None, val_loss_all=None):
        """
        Used for saving the best training model and validation model
        Args:
            model: the NN model
            train_loss: each train epoch loss
            val_loss:each validation epoch loss
        Returns: None
        """
        # saves model weights if loss is the lowest ever
        # Validation sets are used to decide the parameters of the model
        train_min = np.min(train_loss_all)
        if train_loss <= train_min:
            torch.save({"nn_model": model.state_dict()}, savedir+'train_model.pt')
            
        if val_loss is not None and val_loss_all is not None:
            val_min = np.min(val_loss_all)
            if val_loss <= val_min:
                torch.save({"nn_model": model.state_dict()}, savedir+'val_model.pt')

    # plot fig
    # def live_plot(self, subplot=False, **kwargs):
    # """Example for live plot  https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot"""
    #     import matplotlib.pyplot as plt
    #     from collections import deque
    #     import random
    #
    #     # MAX NO. OF POINTS TO STORE
    #     que = deque(maxlen=40)  # list also ok
    #
    #     for i in range(50):
    #         # GENERATING THE POINTS - FOR DEMO
    #         data = random.random()
    #         que.append(data)
    #
    #         # PLOTTING THE POINTS
    #         plt.plot(que)
    #         plt.scatter(range(len(que)), que)
    #
    #         # SET Y AXIS RANGE
    #         plt.ylim(-1, 4)
    #
    #         # DRAW, PAUSE AND CLEAR
    #         plt.gcf().canvas.flush_events()
#             # plt.draw()
#             plt.show(block=False)
#             plt.show(block=False)


    def plotfig_all(self, line_style, colorbar, co2_limit, savedir, savefig=True, **kwargs):
        # each data should include the [tt, Y, U, cost]
        # ---------------------------------------------------------------------
        # Plots and Figures
        # ---------------------------------------------------------------------
        fig = plt.figure(figsize=(12, 6.5), layout='constrained')
        plt.clf()
        spec = fig.add_gridspec(3, 2)

        legend_name = []

        for name, value in kwargs.items():
            legend_name.append(name)
            tt = value[:, 0]
            # Y = value[:, 1:3]
            # U = value[:, 3:5]
            # cost = value[:, 5]


        ax0 = fig.add_subplot(spec[0, 0])
        i = 0
        for name, value in kwargs.items():
            plt.plot(tt, value[:, 1], line_style[i], color=colorbar[i], label='_nolegend_')
            i += 1
        plt.plot(tt, co2_limit, '--', color='#5D565A', label='CO2 release limit')
        plt.legend()
        # plt.legend(['open loop', 'steady-state'])
        plt.xticks(np.arange(np.min(tt), np.max(tt)))
        plt.title("CO2 release")
        plt.xlabel('Time (hour)')
        plt.ylabel('$y_1 \ (Kg/hr)$')
        plt.grid()

        ax1 = fig.add_subplot(spec[0, 1])
        i = 0
        for name, value in kwargs.items():
            plt.plot(tt, value[:, 2], line_style[i], color=colorbar[i])
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt)))
        # plt.legend(['set-point', 'open loop'])
        plt.title("Reboiler temperature")
        plt.ylabel('$y_2 \ (K)$')
        plt.xlabel('Time (hour)')
        plt.grid()

        i = 0

        ax2 = fig.add_subplot(spec[1, 0])
        i = 0
        for name, value in kwargs.items():
            if (i == 0):
                Us = value[:, 3:6]
            if (i == 0) or not (np.array_equal(Us, value[:, 3:6])):  # repeated u not draw
                plt.plot(tt, value[:, 3], line_style[i], color=colorbar[i], drawstyle="steps-post")
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt)))
        plt.title("Liquid input solvent flow rate ($F_L$) ")
        plt.ylabel('$u_1 \ (L/s)$')
        plt.xlabel('Time (hour)')
        plt.grid()

        ax3 = fig.add_subplot(spec[1, 1])
        i = 0
        for name, value in kwargs.items():
            if (i == 0):
                Us = value[:, 3:6]
            if (i == 0) or not (np.array_equal(Us, value[:, 3:6])):  # repeated u not draw
                plt.plot(tt, value[:, 4], line_style[i], color=colorbar[i], drawstyle="steps-post")
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt)))
        plt.title("Fuel flow rate (...)")
        plt.ylabel('$u_2 \ (...)$')
        plt.xlabel('Time (hour)')
        plt.grid()

        ax4 = fig.add_subplot(spec[2, 0])
        i = 0
        for name, value in kwargs.items():
            if (i == 0):
                Us = value[:, 3:6]
            if (i == 0) or not (np.array_equal(Us, value[:, 3:6])):  # repeated u not draw
                plt.plot(tt, value[:, 5], line_style[i], color=colorbar[i], drawstyle="steps-post")
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt)))
        plt.title("Seawater flow rate (...)")
        plt.ylabel('$u_3 \ (...)$')
        plt.xlabel('Time (hour)')
        plt.grid()

        ax5 = fig.add_subplot(spec[2, 1])
        i = 0
        for name, value in kwargs.items():
            plt.plot(tt, value[:, 6], line_style[i], color=colorbar[i])
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt)))
        plt.xlabel('Time (hour)')
        plt.ylabel('Cost ($/hr)')
        plt.title('Economic cost')
        plt.grid()
        # plt.tight_layout()
        # plt.show()

        plt.legend(legend_name)
        if savefig:
            # fig.savefig(savedir + 'result.eps', format='eps', dpi=1200)
            fig.savefig(savedir + 'result.svg', format='svg', dpi=1200)
            fig.savefig(savedir + 'result.pdf', format='pdf', dpi=1200)

    def plotfig_modeling(self, line_style, colorbar, savedir, time_step=40, **kwargs):
        # each data should include the [tt, Y, U, X, Z, P]
        legend_name = []

        for name, value in kwargs.items():
            legend_name.append(name)
            tt = value[:, 0]

        # ------plot measurements and control inputs------
        fig_yu = plt.figure(figsize=(16, 6.5), layout='constrained')
        plt.clf()
        spec = fig_yu.add_gridspec(3, 2)

        ax0 = fig_yu.add_subplot(spec[0, 0])
        i = 0
        for name, value in kwargs.items():
            plt.plot(tt, value[:, 1], line_style[i], color=colorbar[i], label='_nolegend_')
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
        plt.title("CO2 release")
        plt.xlabel('Time (hour)')
        plt.ylabel('$y_1 \ (Kg/hr)$')
        plt.grid()

        ax1 = fig_yu.add_subplot(spec[0, 1])
        i = 0
        for name, value in kwargs.items():
            plt.plot(tt, value[:, 2], line_style[i], color=colorbar[i])
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
        plt.title("Reboiler temperature")
        plt.ylabel('$y_2 \ (K)$')
        plt.xlabel('Time (hour)')
        plt.grid()
        plt.legend(legend_name)

        ax2 = fig_yu.add_subplot(spec[1, 0])
        i = 0
        for name, value in kwargs.items():
            if (i == 0):
                Us = value[:, 3:6]
            if (i == 0) or not (np.array_equal(Us, value[:, 3:6])):  # repeated u not draw
                plt.plot(tt, value[:, 3], line_style[i], color=colorbar[i])
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
        plt.title("Liquid input solvent flow rate ($F_L$) ")
        plt.ylabel('$u_1 \ (L/s)$')
        plt.xlabel('Time (hour)')
        plt.grid()

        ax3 = fig_yu.add_subplot(spec[1, 1])
        i = 0
        for name, value in kwargs.items():
            if (i == 0):
                Us = value[:, 3:6]
            if (i == 0) or not (np.array_equal(Us, value[:, 3:6])):  # repeated u not draw
                plt.plot(tt, value[:, 4], line_style[i], color=colorbar[i])
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
        plt.title("Fuel flow rate  (...)")
        plt.ylabel('$u_2 \ (...)$')
        plt.xlabel('Time (hour)')
        plt.grid()

        ax4 = fig_yu.add_subplot(spec[2, 0])
        i = 0
        for name, value in kwargs.items():
            if (i == 0):
                Us = value[:, 3:6]
            if (i == 0) or not (np.array_equal(Us, value[:, 3:6])):  # repeated u not draw
                plt.plot(tt, value[:, 5], line_style[i], color=colorbar[i])
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
        plt.title("Seawater flow rate (...)")
        plt.ylabel('$u_3 \ (...)$')
        plt.xlabel('Time (hour)')
        plt.grid()

        ax5 = fig_yu.add_subplot(spec[2, 1])
        i = 0
        for name, value in kwargs.items():
            if (i == 0):
                P = value[:, 116:117]
            if (i == 0) or not (np.array_equal(P, value[:, 116:117])):  # repeated u not draw
                plt.plot(tt, value[:, 116], line_style[i], color=colorbar[i])
            i += 1
        plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
        plt.title("Engine load (%)")
        plt.ylabel('$p$ (%)')
        plt.xlabel('Time (hour)')
        plt.grid()

        # fig_yu.savefig(savedir + 'yu.eps', format='eps', dpi=1200)
        fig_yu.savefig(savedir + 'yu.svg', format='svg', dpi=1200)
        fig_yu.savefig(savedir + 'yu.pdf', format='pdf', dpi=1200)

        # ------plot x variables states inputs------
        fig_name = ['x_abs_liq', 'x_abs_gas', 'x_des_liq', 'x_des_gas']
        for m in range(4):
            fig = plt.figure(figsize=(18, 8), layout='constrained')
            phase = ['L', 'G', 'L', 'G']
            for i in range(25):
                item = i//5
                layer = 5 - i%5
                variables_name = ['$C_{N_2, %s%d} \ \ (kmol/m^3)$' % (phase[m], layer),
                                  '$C_{CO_2, %s%d}\ \ (kmol/m^3)$' % (phase[m], layer),
                                  '$C_{MEA, %s%d} \ \ (kmol/m^3)$' % (phase[m], layer),
                                  '$C_{H_2 O, %s%d}\ \ (kmol/m^3)$' % (phase[m], layer),
                                  '$T_{%s%d} \ \ (K)$' % (phase[m], layer)]

                plt.subplot(5, 5, i+1)
                j = 0
                for name, value in kwargs.items():
                    plt.plot(tt, value[:, 6 + m*25 + i], line_style[j], color=colorbar[j])  # value = [tt, Y, U, X] X starts with the 5th
                    j += 1
                plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
                plt.ylabel(variables_name[item])
                plt.xlabel('Time (hour)')
                plt.grid()
            plt.legend(legend_name)
            # fig.savefig(savedir + '%s.eps' % fig_name[m], format='eps', dpi=1200)
            fig.savefig(savedir + '%s.svg' % fig_name[m], format='svg', dpi=1200)
            fig.savefig(savedir + '%s.pdf' % fig_name[m], format='pdf', dpi=1200)

        fig_reb_hex = plt.figure(figsize=(18, 8), layout='constrained')
        for i in range(10):
            variables_name_2 = ['$T_{tube} \ \ (K)$', '$T_{shell}\ \ (K)$', '$T_{reb} \ \ (K)$', '$C_{N_2, Lreb}\ \ (kmol/m^3)$',
                                '$C_{CO_2,Lreb} \ \ (kmol/m^3)$', '$C_{MEA,Lreb}\ \ (kmol/m^3)$', '$C_{H_2 O, Lreb}\ \ (kmol/m^3)$',
                                '$Vapor \ fraction$', '$C_{CO_2, Greb} \ \ (kmol/m^3)$', '$F_{G, reb} \ \ (kmol/s)$']

            plt.subplot(5, 2, i + 1)
            j = 0
            for name, value in kwargs.items():
                plt.plot(tt, value[:, 106 + i], line_style[j], color=colorbar[j])  # value = [tt, Y, U, X] X starts with the 5th
                j += 1
            plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
            plt.ylabel(variables_name_2[i])
            plt.xlabel('Time (hour)')
            plt.grid()
        plt.legend(legend_name)
        # fig_reb_hex.savefig(savedir + 'hex_reb.eps', format='eps', dpi=1200)
        fig_reb_hex.savefig(savedir + 'hex_reb.svg', format='svg', dpi=1200)
        fig_reb_hex.savefig(savedir + 'hex_reb.pdf', format='pdf', dpi=1200)


    def plot_up(self, updata, dt, savedir, time_step=40):
        tt = np.arange(0, len(updata)*dt/3600, dt/3600)
        fig = plt.figure(figsize=(10, 6), layout='constrained')
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plt.plot(tt, updata[:, i])
            plt.xticks(np.arange(np.min(tt), np.max(tt), time_step))
            plt.grid()
            if i == 0:
                plt.title("Liquid input solvent flow rate (F_L) ")
                plt.ylabel('u_1 (L/s)')
                plt.xlabel('Time (hour)')
            if i == 1:
                plt.title("Fuel flow rate (...)")
                plt.ylabel('u_2 (...)')
                plt.xlabel('Time (hour)')
            if i == 2:
                plt.title("Seawater flow rate (...)")
                plt.ylabel('u_3 (...)')
                plt.xlabel('Time (hour)')
            if i == 3:
                plt.title("Engine load (%)")
                plt.ylabel('p (%)')
                plt.xlabel('Time (hour)')
        # fig.savefig(savedir + 'u_p.eps', format='eps', dpi=1200)
        fig.savefig(savedir + 'u_p.svg', format='svg', dpi=1200)
        fig.savefig(savedir + 'u_p.pdf', format='pdf', dpi=1200)





class NNtool():
    def __init__(self, model):
        """
            Neural network related tools
        """
        self.name = 'Xuewen'
        self.tool = MyTool()
        self.model = model
        self.model_param = dict(self.model.named_parameters())
        self.model_param_name = list(self.model_param.keys())

    def save_model(self, savedir, train_loss, val_loss, train_loss_all, val_loss_all):
        """
        Used for saving the best training model and validation model
        Args:
            model: the NN model
            train_loss: each train epoch loss
            val_loss:each validation epoch loss
        Returns: None
        """
        # Validation sets are used to decide the parameters of the model
        train_min = np.min(train_loss_all)
        val_min = np.min(val_loss_all)

        # saves model weights if loss is the lowest ever
        if val_loss <= val_min:
            torch.save({"nn_model": self.model.state_dict()}, savedir+'val_model.pt')
        if train_loss <= train_min:
            torch.save({"nn_model": self.model.state_dict()}, savedir+'train_model.pt')


    def save_parameters(self, model, savedir):
        """
            Save model parameters
            model: neural network model
            savedir: save path for parameters
        """
        self.tool.makedir(savedir)

        # self.model_param = dict(model.named_parameters())
        for name, value in self.model_param.items():
            np.savetxt(savedir + '{}.txt'.format(name), value.detach().numpy())
        print('Model parameters are saved as TXTs.')


    def model_casadi(self, input_dim, activation_fcn=None):
        """
            Construct the well-trained model using casadi and wrap it as a casadi Function for control optimization
            May need to change the structure according to the specific task
            input_dim: the dimension of the input layer
            activation_fcn: 'relu', 'tanh', if None, then no activation functions
            return: self.model_cas a function   output = self.model_cas(input)
        """
        input_var = cs.SX.sym('input', input_dim)
        layer_num = int(len(self.model_param_name)/2)

        input = input_var
        for i in range(layer_num):
            fc_weight = np.array(self.model_param[self.model_param_name[i]].detach().numpy())
            fc_bias = np.array(self.model_param[self.model_param_name[i+1]].detach().numpy())
            output = cs.mtimes(fc_weight, input) + fc_bias
            if i != layer_num-1:                        # last layer no activation function
                if activation_fcn == 'relu':            # different activation functions
                    output = cs.fmax(output)
                elif activation_fcn == 'tanh':
                    output = cs.tanh(output)
            input = output                              # update input for the next layer

        # Create casadi function
        self.model_cas = cs.Function('model', [input_var], [output])





class MyThread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.results = None

    def run(self):
        self.results = self.func(*self.args)

    def getResult(self):
        return self.results








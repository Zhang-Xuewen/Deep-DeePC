"""
Name: main.py
Author: Xuewen Zhang
Date: 18/04/2024
Project: DeepDeePC
"""
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
import argparse

import DDeePC as ddpc
from ddeepc_args import update_variables, build_args
from QYtool import timer, datatool, dirtool, rprint, progressbar



def generate_setpoints(N_sp=100, system='threetanks') -> None:
    """ 
    Genearte different set-points for the WWTPs
        system: 'WWTPs' or 'threetanks' or 'siso' or 'grn'  
    """
    sp_dir = f'./_data/{system}/setpoints/'
    dirtool.makedir(sp_dir)
    if system == 'WWTPs':
        x0, p = datatool.loadtxt(f'./_data/{system}/', 'ss_open', 'inf_rain_mean') 
        plant = ddpc.waste_water_system(x0, p) 
        action_low = np.array([30, 5000])
        action_high = np.array([210, 5 * 18446-5000])
    elif system == 'threetanks':
        plant = ddpc.three_tank_system()
        action_low = plant.action_low
        action_high = plant.action_high
    elif system == 'siso':
        plant = ddpc.siso()
        action_low = plant.action_low
        action_high = plant.action_high
    elif system == 'grn':
        plant = ddpc.grn()
        action_low = plant.action_low
        action_high = plant.action_high

    action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    
    probar = progressbar.probar1()
    with probar:
        task = probar.add_task('Generate set-points', total=N_sp, unit='sp')
        for j in range(N_sp):
            plant.reset()
            us = action_space.sample()
            for i in range(10000):
                plant.step(us, p_constant=True) if system == 'WWTPs' else plant.step(us)
            xs = plant.state_buffer.memory[-1]
            us = plant.state_buffer.memory_action[-1]
            ys = xs[plant.observed_dims]
            
            exp, exp_dir = dirtool.create_new_folder_with_max_number(sp_dir)
            datatool.savetxt(exp_dir+'/', xs=xs, us=us, ys=ys)
            probar.update(task, advance=1)
    return 


def test_deepc(args):
    """ Test the parameters for DeePC """
    rprint("[green]Verify if the parameters for DeePC works well ...[/green]")
    test_args = args.copy()
    test_args['N'] = 200
    deepc = ddpc.deepc(**test_args)
    deepc.test()
    rprint(f":warning: [red]Please check the performance in[/red] {args['fig_dir']} [red]and then decide to train the model or not ...[/red]")
    return 


def learn(args)->None:
    test_deepc(args)
    rprint("[green]Neural network training ...[/green]")
    train_fcn = ddpc.train(args)
    train_fcn.run()


def control(exp, system='threetanks', test_num=100, N=None, x0_std_dev=None, noise=None, con_opt=None) -> None:
    # some times variance too large, the integrator will failed, then retry
    try:
        print(f"{'-'*30}Control{'-'*30}\n")   
        args = datatool.load_config(f"./_results/{system}/{exp}/", 'args')
        if N is not None:
            args['N'] = N
        if x0_std_dev is not None:
            args['x0_std_dev'] = x0_std_dev
        if noise is not None:
            args['noise'] = noise
        if con_opt is not None:
            args['con_opt'] = con_opt
        controller = ddpc.control(args)
        controller.rollout(test_num)
    except Exception as e:
        print(f"{'-'*30}Solver failed, retry{'-'*30}\n")
        return control(exp, system, test_num, N, x0_std_dev, noise, con_opt)
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepDeePC')
    parser.add_argument('--gen_sp', type=int, default=0, help='Generate set-points for selected system')
    parser.add_argument('--system', type=str, default='threetanks', help='System name: WWTPs, threetanks, siso, grn')
    parser.add_argument('--train', action='store_true', default=False, help='Start a nex exp for training')
    parser.add_argument('--test', type=int, default=0, help='Test the control performance of the specific exp')
    parser.add_argument('--test_num', type=int, default=1, help='Number of numbers tested for control performance')
    parser.add_argument('--N', type=int, default=100, help='Time steps for one set-point')
    parser.add_argument('--noise', action='store_true', default=False, help='Add noise to the system')
    par_args = parser.parse_args()

    # Load the variables
    system = par_args.system
    test_num = par_args.test_num
    N = par_args.N
    N_sp = par_args.gen_sp
    noise = par_args.noise
    
    if system not in ['WWTPs', 'threetanks', 'siso', 'grn']:
        raise ValueError("System name is not correct, please check it. (System name: WWTPs, threetanks, siso, grn)")
    
    ## ----- Generate set-points for system -----
    if par_args.gen_sp != 0:
        generate_setpoints(N_sp=N_sp, system=system)
    
    ## ----- Learn the g by deep learning -----
    ## load args
    if par_args.train:
        args_fcn = build_args(**update_variables)
        args = args_fcn.args
        learn(args) 
        exp = args['exp']      # exp number that used to control
        system = args['system']    
        control(exp, system=system, test_num=test_num, N=N, noise=noise, con_opt=False)
        control(exp, system=system, test_num=test_num, N=N, noise=noise, con_opt=True)
    
    ## ----- Implement DeePC via learned g -----
    if par_args.test != 0:
        exp = par_args.test
        control(exp, system=system, test_num=test_num, N=N, noise=noise, con_opt=False)
        control(exp, system=system, test_num=test_num, N=N, noise=noise, con_opt=True)
    

        
    
    

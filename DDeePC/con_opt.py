"""
Name: con_opt.py
Author: Xuewen Zhang
Date: 20/06/2024
Project: DeepDeePC
Description: this script is the optimization for ensure the constraints of the system
"""

import time
import numpy as np
import casadi as cs
import casadi.tools as ctools
from rich import print as rprint

def timer(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        rprint(f":stopwatch:  Function: [cyan]{f.__name__}[/cyan] ==> Time elapsed: [orange]{end-start:.3f}[/orange] s")
        return ret
    return wrapper


class con_opt():
    def __init__(self, g_dim, Np, Uf, Yf, ineqconidx=None, ineqconbd=None):
        """ Initialize the variables
        Args: 
                 g_dim: [int]             | the dimension of the operator g
                    Np: [int]             | the number of the prediction horizon
                    Uf: [np.array]        | the Hankel matrix for the input u
                    Yf: [np.array]        | the Hankel matrix for the output y
            ineqconidx: [dict|[str,list]] |  specify the wanted constraints for u and y, if None, no constraints
                                          |      e.g., only have constraints on u2, u3, {'u': [1,2]}; 'y' as well
             ineqconbd: [dict|[str,list]] |  specify the bounds for u and y, should be consistent with "ineqconidx"
                                          |      e.g., bound on u2, u3, {'lbu': [1,0], 'ubu': [10,5]}; lby, uby as well
                                          
        """
        self.g_dim = g_dim
        self.Np = Np
        self.Uf = Uf
        self.Yf = Yf
        self.u_dim = int(Uf.shape[0]/Np)
        self.y_dim = int(Yf.shape[0]/Np)
        self.flag = {'init_solver': False}
        
        self.optimizing_target, self.parameters = self._init_variables()
        self.Hc, self.lbc_ineq, self.ubc_ineq = self._init_ineq_cons(ineqconidx, ineqconbd)
        return
        
        
    def _init_variables(self):
        """ Initialize the variables """
        optimizing_target = ctools.struct_symSX([
            ctools.entry('g', shape=tuple([self.g_dim, 1]))
        ])
        
        parameters = ctools.struct_symSX([
            ctools.entry('g_ref', shape=tuple([self.g_dim, 1]))
        ])
        return optimizing_target, parameters
    
    
    def _init_ineq_cons(self, ineqconidx=None, ineqconbd=None):
        """
            Obtain Hankel matrix that used for the inequality constrained variables
                           lbc <= Hc * g <= ubc
            return  Hc, lbc, ubc
        """
        if ineqconidx is None:
            print(">> DeePC design have no constraints on 'u' and 'y'.")
            Hc, lbc, ubc = [], [], []
        else:
            Hc_list = []
            lbc_list = []
            ubc_list = []
            for varname, idx in ineqconidx.items():
                if varname == 'u':
                    H_all = self.Uf.copy()
                    dim = self.u_dim
                    lb = ineqconbd['lbu']
                    ub = ineqconbd['ubu']
                elif varname == 'y':
                    H_all = self.Yf.copy()
                    dim = self.y_dim
                    lb = ineqconbd['lby']
                    ub = ineqconbd['uby']
                else:
                    raise ValueError("%s variable not exist, should be 'u' or/and 'y'!" % varname)

                ## TODO: constraint on Np steps
                idx_H = [v + i * dim for i in range(self.Np) for v in idx]
                Hc_list.append(H_all[idx_H, :])
                lbc_list.append(np.tile(lb, self.Np))
                ubc_list.append(np.tile(ub, self.Np))
                ## TODO: constraint on m step
                # m = 3
                # idx_H = [v + i * dim for i in range(m) for v in idx]
                # Hc_list.append(H_all[idx_H, :])
                # lbc_list.append(np.tile(lb, m))
                # ubc_list.append(np.tile(ub, m))

            Hc = np.concatenate(Hc_list)
            lbc = np.concatenate(lbc_list).flatten().tolist()
            ubc = np.concatenate(ubc_list).flatten().tolist()
        return Hc, lbc, ubc
    
    
    @timer
    def init_solver(self, solver='ipopt', opts={}):
        """ Initialize the solver 
        optimization problem:
                        minimize: || g - g_ref  ||_2^2
                        subject to: lbc <= Hc * g <= ubc
        solver: [str] | the solver used for optimization, default is 'ipopt'
        opts: [dict]  | the config of the solver; max iteration, print level, etc.
                e.g.:         opts = {
                                        'ipopt.max_iter': 100,  # 50
                                        'ipopt.tol': 1e-5,
                                        'ipopt.print_level': 1,
                                        'print_time': 0,
                                        # 'ipopt.acceptable_tol': 1e-8,
                                        # 'ipopt.acceptable_obj_change_tol': 1e-6,
                                    }
        """
        g, = self.optimizing_target[...]
        g_ref, = self.parameters[...]
        
        # objective function
        J = cs.mtimes((g - g_ref).T, (g - g_ref))
        
        # constraints
        C = []
        lbc, ubc = [], []
        
        # inequality constraints
        C += [cs.mtimes(self.Hc, g)]
        lbc.extend(self.lbc_ineq)
        ubc.extend(self.ubc_ineq)
        
        # formulate the optimization problem
        opt_prob = {'f': J, 'x': self.optimizing_target, 'p': self.parameters, 'g': cs.vertcat(*C)}
        
        self.solver = cs.nlpsol('solver', solver, opt_prob, opts)
        self.lbc = lbc
        self.ubc = ubc
        self.flag['init_solver'] = True
        return 
    
    
    def solver_step(self, g_ref):
        """ Solve the optimization problem for one time 
        Args:
            g_ref: [np.array][g_dim, 1]] | the reference of the operator g
        """
        if not self.flag['init_solver']:
            raise ValueError("Solver not initialized, please initialize the solver first!")
        
        t_ = time.time()
        sol = self.solver(x0=g_ref, p=g_ref, lbg=self.lbc, ubg=self.ubc)
        t_s = time.time() - t_
        
        g_opt = sol['x'].full().ravel()
        return g_opt.reshape(-1, 1), t_s
    
        
        
        
        
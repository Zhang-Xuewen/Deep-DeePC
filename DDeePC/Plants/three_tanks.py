"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt

class three_tank_system(gym.Env):

    def __init__(self, x0_std_dev=0.1, us= np.array([2.87e9, 1.0e9, 2.87e9]), xs=np.array([1.920462734802070026e-01, 6.753561181469071029e-01, 4.768612896432074422e+02, 2.116736585616537936e-01, 6.560585575539794601e-01, 4.695016333820219074e+02, 7.212661966828316784e-02, 6.895290277477214014e-01, 4.715140681309391084e+02])):
        self.t = 0
        self.action_sample_period = 80
        self.sampling_period = 0.025  # 0.005 hour
        self.h = 0.001
        self.sampling_steps = int(self.sampling_period/self.h)
        self.delay = 0
        self.observed_dims = [2, 5, 8]
        self.x0_std_dev = x0_std_dev
        
        self.x_dim = 9
        self.y_dim = 3
        self.u_dim = 3
        self.p_dim = 0

        self.s2hr = 3600
        self.MW = 250e-3
        self.sum_c = 2E3
        self.T10 = 300
        self.T20 = 300
        self.F10 = 5.04
        self.F20 = 5.04
        self.Fr = 50.4
        self.Fp = 0.504
        self.V1 = 1
        self.V2 = 0.5
        self.V3 = 1
        self.E1 = 5e4
        self.E2 = 6e4
        self.k1 = 2.77e3 * self.s2hr
        self.k2 = 2.6e3 * self.s2hr
        self.dH1 = -6e4 / self.MW
        self.dH2 = -7e4 / self.MW
        self.aA = 3.5
        self.aB = 1
        self.aC = 0.5
        self.Cp = 4.2e3
        self.R = 8.314
        self.rho = 1000
        self.xA10 = 1
        self.xB10 = 0
        self.xA20 = 1
        self.xB20 = 0
        self.Hvap1 = -35.3E3 * self.sum_c
        self.Hvap2 = -15.7E3 * self.sum_c
        self.Hvap3 = -40.68E3 * self.sum_c

        self.kw = np.array([1, 1, 5, 1, 1, 5, 1, 1, 5]) # noise deviation
        self.bw = np.array([0.5, 0.5, 5, 0.5, 0.5, 5, 0.5, 0.5, 5]) # noise bound
        self.noise = None
        # self.kw = np.array([0.1, 0.1, 5, 0.1, 0.1, 5, 0.1, 0.1, 5]) # noise deviation
        # self.bw = np.array([5, 5, 15, 5, 5, 15, 5, 5, 15]) # noise bound

        self.xs = xs  # np.array([0.1763, 0.6731, 480.3165, 0.1965, 0.6536, 472.7863, 0.0651, 0.6703, 474.8877])
        self.ys = xs[[2, 5, 8]]  # np.array([480.3165, 472.7863, 474.8877])
        self.us = us  # 1.12 * np.array([2.9e9, 1.0e9, 2.9e9])

        high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

        self.action_low = np.array([2.8e9, 0.9e9, 2.8e9], dtype=np.float32)        # 0.2 * self.us   # todo: check u limit
        self.action_high = np.array([3e9, 1.2e9, 3e9], dtype=np.float32)           # 1.5 * self.us
        self.y_low = np.array([0, 0, 0])                         # set by myself, just for test, not real bound
        self.y_high = np.array([700, 700, 700])                  # set by myself, just for test, not real bound
        
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high)
        self.observation_space = spaces.Box(-high, high) # not used

        self.seed()
        self.state_buffer = state_buffer(self.delay)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action, noise=False):
        action = np.clip(action, self.action_low, self.action_high)
        
        x0 = self.state
        for i in range(self.sampling_steps):
            process_noise = np.random.normal(np.zeros_like(self.kw), self.kw) \
                if self.noise is None else self.noise[self.t*self.sampling_steps + i]
            process_noise = np.clip(process_noise, -self.bw, self.bw)
            x0 = x0 + self.derivative(x0, action)*self.h 
            x0 = x0 + process_noise*self.h if noise else x0
            
        self.state = x0
        self.t += 1
        self.state_buffer.memorize(self.state)                              # save the state trajectories
        self.state_buffer.memorize_u(action)                                # save the action trajectories
        
        cost = np.linalg.norm(self.state - self.xs)
        done = False
        data_collection_done = False

        return x0, cost, done, dict(reference=self.xs, data_collection_done=data_collection_done)


    def reset(self, test=False, seed_=1):
        self.state_buffer.reset()
        self.a_holder = self.action_space.sample()
        self.state = self.xs + np.random.normal(np.zeros_like(self.xs), self.xs*self.x0_std_dev)
        self.state_buffer.memorize(self.state)  # save the initial state
        if test:
            np.random.seed(seed_)
            self.state = [0.9599, 0.9039, 1.1200, 0.9726, 1.1643, 0.8727, 0.9055, 0.8582, 0.8544] * self.xs
        self.noise = None
        self.t = 0
        self.time = 0
        return self.state
    
    
    def set_initial(self, x0=None, noise=None):
        """ Set the initial state of the system, and the given noise"""
        self.reset()
        self.state = x0 if x0 is not None else self.state
        self.noise = noise if noise is not None else self.noise
        self.state_buffer.reset()
        self.state_buffer.memorize(self.state)  # save the initial state
        self.t = 0
        self.time = 0
        return
    
    
    def generate_noise(self, N):
        """ Generate the noise in advance 
        Args:
            N (int): the number of noise steps
        return:
            noise (array)(N*sampling_steps,): the noise array
            1 step will have sampling_steps noise
        """
        noise = []
        for i in range(N*self.sampling_steps):
            process_noise = np.random.normal(np.zeros_like(self.kw), self.kw)
            process_noise = np.clip(process_noise, -self.bw, self.bw)
            noise.append(process_noise)
        return np.array(noise)

    
    def derivative(self, x, us):
        xA1 = x[0]
        xB1 = x[1]
        T1 = x[2]

        xA2 = x[3]
        xB2 = x[4]
        T2 = x[5]

        xA3 = x[6]
        xB3 = x[7]
        T3 = x[8]

        Q1 = us[0]
        Q2 = us[1]
        Q3 = us[2]

        xC3 = 1 - xA3 - xB3
        x3a = self.aA * xA3 + self.aB * xB3 + self.aC * xC3

        xAr = self.aA * xA3 / x3a
        xBr = self.aB * xB3 / x3a
        xCr = self.aC * xC3 / x3a

        F1 = self.F10 + self.Fr
        F2 = F1 + self.F20
        F3 = F2 - self.Fr - self.Fp

        f1 = self.F10 * (self.xA10 - xA1) / self.V1 + self.Fr * (xAr - xA1) / self.V1 - self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1
        f2 = self.F10 * (self.xB10 - xB1) / self.V1 + self.Fr * (xBr - xB1) / self.V1 + self.k1 * np.exp(-self.E1 / (self.R * T1)) * xA1 - self.k2 * np.exp(
            -self.E2 / (self.R * T1)) * xB1
        f3 = self.F10 * (self.T10 - T1) / self.V1 + self.Fr * (T3 - T1) / self.V1 - self.dH1 * self.k1 * np.exp(
            -self.E1 / (self.R * T1)) * xA1 / self.Cp - self.dH2 * self.k2 * np.exp(
            -self.E2 / (self.R * T1)) * xB1 / self.Cp + Q1 / (self.rho * self.Cp * self.V1)

        f4 = F1 * (xA1 - xA2) / self.V2 + self.F20 * (self.xA20 - xA2) / self.V2 - self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2
        f5 = F1 * (xB1 - xB2) / self.V2 + self.F20 * (self.xB20 - xB2) / self.V2 + self.k1 * np.exp(-self.E1 / (self.R * T2)) * xA2 - self.k2 * np.exp(
            -self.E2 / (self.R * T2)) * xB2
        f6 = F1 * (T1 - T2) / self.V2 + self.F20 * (self.T20 - T2) / self.V2 - self.dH1 * self.k1 * np.exp(
            -self.E1 / (self.R * T2)) * xA2 / self.Cp - self.dH2 * self.k2 * np.exp(
            -self.E2 / (self.R * T2)) * xB2 / self.Cp + Q2 / (self.rho * self.Cp * self.V2)

        f7 = F2 * (xA2 - xA3) / self.V3 - (self.Fr + self.Fp) * (xAr - xA3) / self.V3
        f8 = F2 * (xB2 - xB3) / self.V3 - (self.Fr + self.Fp) * (xBr - xB3) / self.V3
        f9 = F2 * (T2 - T3) /self.V3 + Q3 / (self.rho * self.Cp * self.V3) + (self.Fr + self.Fp) * (xAr * self.Hvap1 + xBr * self.Hvap2 + xCr * self.Hvap3) / (
                self.rho * self.Cp * self.V3)

        F = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9])
        
        return F
    
    def render(self, mode='human'):

        return

    def get_action(self, noise=True):
        # every action_sample_period, generate a new action u
        if self.t % self.action_sample_period == 0:
            self.a_holder = self.action_space.sample()
        a = self.a_holder + np.random.normal(np.zeros_like(self.us), self.us*0.01) if noise else self.a_holder
        a = np.clip(a, self.action_low, self.action_high)

        return a

    def get_action_pid(self):
        """
            Generate new action based on the PID control
            u: [u1, u2, u3], u1: KLa5, and u2: Qa
            y: [x3, x6, x9], temperatures of three tanks
            action: u at time instant k-1
            return: action u at time instant k
        """
        # first step, the u set to us
        if len(self.state_buffer.memory_action) == 0:
            action = self.us.copy()
        else:
            action = self.state_buffer.memory_action[-1].copy()                   # u at time instant k-1

        # first step, the y(k-1) set to ys
        yk = self.state_buffer.memory[-1][self.observed_dims]                     # y at time instant k
        if len(self.state_buffer.memory) == 1:
            yk_1 = self.ys.copy()                                                 # y at time instant k-1
        else:
            yk_1 = self.state_buffer.memory[-2][self.observed_dims].copy()        # y at time instant k-1

        # PID for u1 based on y1: x3
        K_1 = 90000                                                          # m3 / d / (g N/m3);
        Ti_1 = 1                                                     # days
        T_1 = 0.005
        Ki_1 = 0.004
        ek_1 = self.ys[0] - yk[0]                                         # e(k)
        ek_1_1 = self.ys[0] - yk_1[0]                                     # e(k - 1)
        # du1 = K_1 * ((ek_1 - ek_1_1) + Ki_1 * ek_1 * self.sampling_period + T_1 / Ti_1 * ek_1)
        du1 = K_1 * (ek_1 + Ki_1 * ek_1 * self.sampling_period)

        # PID for u2 based on y2: x6
        K_2 = 70000                                                      # m3 / d / (g N/m3);
        Ti_2 = 1                                                     # days
        T_2 = 0.005                                                     # days
        Ki_2 = 0.002
        ek_2 = self.ys[1] - yk[1]                                        # e(k)
        ek_1_2 = self.ys[1] - yk_1[1]                                    # e(k - 1)
        # du2 = K_2 * ((ek_2 - ek_1_2) + Ki_2 * ek_2 * self.sampling_period + T_2 / Ti_2 * ek_2)
        du2 = K_2 * (ek_2 + Ki_2 * ek_2 * self.sampling_period)

        # PID for u3 based on y3: x9
        K_3 = 100000                                                      # m3 / d / (g N/m3);
        Ti_3 = 1                                                     # days
        T_3 = 0.005
        Ki_3 = 0.004                                                          # days
        ek_3 = self.ys[2] - yk[2]                                        # e(k)
        ek_1_3 = self.ys[2] - yk_1[2]                                    # e(k - 1)
        # du3 = K_3 * ((ek_3 - ek_1_3) + Ki_3 * ek_3 * self.sampling_period + T_3 / Ti_3 * ek_3)
        du3 = K_3 * (ek_3 + Ki_3 * ek_3 * self.sampling_period)

        # action at current time instant
        action[0] = action[0] + du1
        action[1] = action[1] + du2
        action[2] = action[2] + du3

        action = np.clip(action, self.action_low, self.action_high)
        return action


    def get_noise(self):
        scale = 0.1 * self.xs
        return np.random.normal(np.zeros_like(self.xs), scale)


class state_buffer(object):

    def __init__(self, delay):

        self.delay = delay
        self.memory = []             # state trajectory
        self.memory_action = []      # action trajectory

    def memorize(self, s):

        self.memory.append(s)
        return

    def memorize_u(self, u):

        self.memory_action.append(u)
        return

    def get_state(self, t):

        if t < self.delay:
            return None
        else:
            return self.memory[t]

    def get_action(self, t):

        if t < self.delay:
            return None
        else:
            return self.memory_action[t]

    def reset(self):
        self.memory = []
        self.memory_action = []


if __name__=='__main__':

    env = three_tank_system()
    T = 100
    env.reset()

    for i in range(int(T)):
        # action = env.us
        action = env.get_action(noise=False)
        _ = env.step(action, noise=False)
        
    x = np.stack(env.state_buffer.memory[:-1])
    u = np.stack(env.state_buffer.memory_action)
    t = np.arange(T)*env.sampling_period
    
    fig, ax = plt.subplots(ncols=3, nrows=3, sharex=True, figsize=(12, 9))
    for i in range(env.x_dim):
        ax[i//3, i%3].plot(t, x[:, i], color='blue')
        ax[i//3, i%3].set_ylabel(f'x{i+1}')
        ax[i//3, i%3].set_xlabel('Time (hour)')
    plt.savefig('x.pdf')
    # plt.clf()
    
    fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(12, 3))
    for i in range(env.u_dim):
        ax[i].plot(t, u[:,i])
        ax[i].set_ylabel(f'u{i+1}')
        ax[i].set_xlabel('Time (hour)')
    plt.show()
    plt.savefig('u.pdf')
    
    # np.savetxt('x.txt', x)
    # np.savetxt('u.txt', u)
    # np.savetxt('y.txt', x[:,env.observed_dims])

 







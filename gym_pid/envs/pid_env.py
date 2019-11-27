import PID
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

import gym
import numpy as np
import random
import pandas as pd
import cv2
from IPython.display import clear_output
from gym import spaces
from matplotlib import pyplot as plt

# Each state is an image. State space is 2D.

NUM_OF_ACTIONS = 7
INPUT_SHAPE = (3,)

def test_pid(P=0.2, I=0.0, D=0.0, L=50):
    """Self-test PID class
    .. note::
        ...
        for i in range(1, END):
            pid.update(feedback)
            output = pid.output
            if pid.SetPoint > 0:
                feedback += (output - (1/i))
            if i>9:
                pid.SetPoint = 1
            time.sleep(0.02)
        ---
    """
    pid = PID.PID(P, I, D)

    pid.SetPoint = 0.0
    pid.setSampleTime(0.01)

    END = L
    feedback = 0

    feedback_list = []
    time_list = []
    setpoint_list = []

    for i in range(1, END):
        pid.update(feedback)
        output = pid.output
        if pid.SetPoint > 0:
            feedback += (output - (1 / i))
        if i > 9:
            pid.SetPoint = 1
        time.sleep(0.02)

        feedback_list.append(feedback)
        setpoint_list.append(pid.SetPoint)
        time_list.append(i)

    error = np.asarray(setpoint_list) - np.asarray(feedback_list)
    #error = np.clip(error, -50, 50)
    error_std = np.std(error)
    return error_std

class pidEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(NUM_OF_ACTIONS)
        self.observation_space = spaces.Box(low=-100, high=100, shape=INPUT_SHAPE, dtype='float64')
        self.Kp = 0.4
        self.Ki = 5
        self.Kd = 0.05
        self.state = np.array([self.Kp, self.Ki, self.Kd])
        self.prev_error = 0.0
        self.new_error = 0.0
        self.done = False

    def step(self, action, update=0):
        
        temp_Kp = self.Kp
        temp_Ki = self.Ki
        temp_Kd = self.Kd
        
        if action == 0:  # increase P
            temp_Kp = 1.1*self.Kp
        elif action == 1:  # decrease P
            temp_Kp = 0.9*self.Kp
        elif action == 2:  # increase I
            temp_Ki = 1.1*self.Ki
        elif action == 3:  # decrease I
            temp_Ki = 0.9*self.Ki
        elif action == 4:  # increase D
            temp_Kd = 1.1*self.Kd
        elif action == 5:  # decrease D
            temp_Kd = 0.9*self.Kd
        
        self.new_error = test_pid(temp_Kp, temp_Ki, temp_Kd, 20)
        reward = (self.prev_error - self.new_error) > 0

        self.state = np.array([temp_Kp, temp_Ki, temp_Kd])
        
        if update == 1:
            self.prev_error = self.new_error
            self.done = self.prev_error < 0.15
            self.Kp = temp_Kp
            self.Ki = temp_Ki
            self.Kd = temp_Kd
            
        if self.done:
            print('P: ', self.Kp, 'I: ', self.Ki, 'D: ', self.Kd)
            

        return self.state, reward, self.done, {}

    def reset(self):
        self.Kp = 0.4
        self.Ki = 5
        self.Kd = 0.05
        self.state = np.array([self.Kp, self.Ki, self.Kd])
        self.prev_error = 0.0
        self.new_error = 0.0
        self.done = False
        return self.state

    def render(self, mode='human'):
        print('Error Std. Dev.: ', self.new_error)



"""
config.py

集中管理 MPC 相关设置(Q, R, P, Horizon, Ts 等）
以及初始状态采样与随机初始猜测生成函数等。
"""
import  numpy as np
import random


Horizon = 128          # 预测步数
Ts = 0.01             # 采样时间
Num_State = 14         #  状态维数（位置 + 速度）
Num_Q = 7         # 关节数目
Num_Velocity = 7         # 关节速度维数
Num_P = 3         # 末端执行器位置维数
Num_Input = 7         # 控制量维数（推力）
gravity = [0, 0, -9.81] # 重力加速度
root = "panda_link0"
tip = "panda_link8"
tau_max = 100.0 # 最大关节力矩

# 状态和控制量的权重矩阵
Q= np.eye(3)*1000 #把前 7 个状态(关节角度)的对角权重调成 10
# idx = np.arange(3)
# Q[idx,idx]= 100.0

R = np.eye(7) * 0.01

P = Q



def GenerateRandomInitialGuess(min_random=-20, max_random=20):
    
    u_guess_4 = np.round(random.uniform(min_random, max_random),2)
    u_guess_5 = np.round(random.uniform(min_random, max_random),2)
    u_guess_7 = np.round(random.uniform(min_random, max_random),2)
    u_ini_guess= np.array([0,0,0,u_guess_4,u_guess_5,0,u_guess_7])

    return u_ini_guess
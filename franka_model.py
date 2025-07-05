"""
cartpole_model.py

定义 CartPole + 虚拟状态 (theta_stat) 的动力学模型，
供 ACADOS OCP 使用。
"""

import numpy as np
import casadi as ca
import pinocchio.casadi as cpin
import pinocchio as pin
from acados_template import AcadosModel
import config
import os

def export_franka_ode_model():
    """
    状态: q : 7个关节角度 + 7个关节角速度
    控制: tau : 7个关节力矩
    机械臂franka_emika_panda的动力学模型
    
    """

    urdf_path = "panda_arm.urdf"
    model = pin.buildModelFromUrdf(urdf_path)

    # 2. 转成 CasADi 标量模型，并创建配套 data
    cmodel = cpin.Model(model)                   # SX 标量模型
    cdata  = cmodel.createData()
    # 常量定义
    nx = config.Num_State  # 状态数量
    nq = config.Num_Q  # 关节数量
    nv = config.Num_Velocity  # 关节速度数量
    nu = config.Num_Input  # 控制输入数量
    x_sym = ca.SX.sym('x', nx)       # [q(7), qdot(7)]
    u_sym = ca.SX.sym('u', nu)         # [tau(7)]
    xdot_sym = ca.SX.sym('xdot', nx) # [qdot(7), qddot(7)]

    
    q     = x_sym[:7] # 关节角度
    qdot  = x_sym[7:14]  # 关节角速度
    tau   = u_sym   # 关节力矩

    qddot = cpin.aba(cmodel, cdata, q, qdot, tau)
    
    # 运动学作为输出，不作为状态
    cpin.framesForwardKinematics(cmodel, cdata, q)
    ee_id  = cmodel.getFrameId(config.tip)
    T_ee   = cdata.oMf[ee_id]                      # SE3: rotation+translation
    p_ee   = T_ee.translation  
    p_expr = ca.Function("fk_pos", [q], [p_ee])
    
    f_expl = ca.vertcat(qdot, qddot)
    f_impl = xdot_sym - f_expl

    model = AcadosModel()
    model.name = 'franka_14dof'
    model.x = x_sym
    model.xdot = xdot_sym
    model.u = u_sym
    model.p = []
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.cost_y_expr = ca.vertcat(p_ee, tau) 
    model.cost_y_expr_e = ca.vertcat(p_ee)     
    
    return model

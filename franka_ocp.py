import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import config  # 引用 config.py
import scipy.linalg
import time
# 导入 CartPole 模型
from franka_model import export_franka_ode_model

def clear_solver_state(ocp_solver, N_horizon):
    # 清空状态和控制输入
    for i in range(N_horizon):
        ocp_solver.set(i, "x", np.zeros_like(ocp_solver.get(i,"x")))
        ocp_solver.set(i, "u", np.zeros_like(ocp_solver.get(i,"u")))
    ocp_solver.set(N_horizon, "x", np.zeros_like(ocp_solver.get(N_horizon,"x")))

def get_guess_from_solver_result(ocp_solver, N_horizon):
    u_guess = np.zeros((config.Num_Input, N_horizon))
    x_guess = np.zeros((config.Num_State, N_horizon+1))
    for i in range(N_horizon-1):
        u_guess[:, i] = ocp_solver.get(i+1, "u")
        x_guess[:, i] = ocp_solver.get(i+1, "x")
    u_guess[:, N_horizon-1] = ocp_solver.get(N_horizon-1, "u")
    x_guess[:, N_horizon-1] = ocp_solver.get(N_horizon, "x")
    x_guess[:, N_horizon] = ocp_solver.get(N_horizon, "x")
    return u_guess, x_guess

def create_ocp_solver(x0):
    ocp = AcadosOcp()

    # Parameter loading
    Nx = config.Num_State
    Nu = config.Num_Input
    N  = config.Horizon
    tf = N * config.Ts   # 总时域 tf = N_horizon * Ts

    #  OCP setup
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = tf


    model = export_franka_ode_model()
    ocp.model = model
    ocp.model.x = model.x
    ocp.model.u = model.u

    # 成本函数
    p_target = np.array([0.3,0.3,0.5  ]) 
    u_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = model.cost_y_expr
    ocp.cost.W = scipy.linalg.block_diag(config.Q, config.R)
    ocp.cost.yref = np.concatenate((p_target, u_target))
    ocp.dims.ny = ocp.cost.yref.shape[0]
    

    # 终端成本
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = model.cost_y_expr_e
    ocp.cost.W_e = config.P
    ocp.cost.yref_e = p_target
    ocp.dims.ny_e = ocp.cost.yref_e.shape[0]
    
    # Constraints (from panda_arm.urdf)
    # 关节角度限制（单位：弧度）
    # panda_joint1: [-2.8973, 2.8973]
    # panda_joint2: [-1.7628, 1.7628]
    # panda_joint3: [-2.8973, 2.8973]
    # panda_joint4: [-3.0718, -0.0698]
    # panda_joint5: [-2.8973, 2.8973]
    # panda_joint6: [-0.0175, 3.7525]
    # panda_joint7: [-2.8973, 2.8973]
    q_min = np.array([
        -2.8973,
        -1.7628,
        -2.8973,
        -3.0718,
        -2.8973,
        -0.0175,
        -2.8973
    ])
    q_max = np.array([
        2.8973,
        1.7628,
        2.8973,
        -0.0698,
        2.8973,
        3.7525,
        2.8973
    ])
    # 只对前7维q加约束
    ocp.constraints.idxbx = np.arange(7)
    ocp.constraints.lbx = q_min
    ocp.constraints.ubx = q_max

    ocp.constraints.x0 = x0


    # 求解器设置

    # ocp.solver_options.integrator_type = 'ERK' 

    # ocp.solver_options.nlp_solver_tol_stat = 1e-4  
    # ocp.solver_options.levenberg_marquardt = 1e-1 
    # ocp.solver_options.print_level = 0  
    # # ocp.solver_options.qp_solver_iter_max = 50
    # # ocp.solver_options.globalization = 'FIXED_STEP' 
    # ocp.solver_options.regularize_method = 'CONVEXIFY' 
    # # ocp.solver_options.eps_regularization = 1e-3  


    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES' 
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.integrator_type    = 'ERK'   # 显式 RK
    ocp.solver_options.sim_method_num_stages = 4    # 4 个 stage = RK4
    # ocp.solver_options.sim_method_order      = 4    # 四阶
    ocp.solver_options.sim_method_num_steps  = 10
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.nlp_solver_max_iter = 150
    ocp.solver_options.nlp_solver_tol_stat = 5e-3
    ocp.solver_options.levenberg_marquardt = 1.0

    # 构造 OCP 求解器
    isbuild = True
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_double_pendulum.json", generate= isbuild, build = isbuild)

    acados_integrator = AcadosSimSolver(ocp, json_file = "acados_ocp_double_pendulum.json", generate = isbuild, build = isbuild)

    return ocp, acados_solver, acados_integrator


def simulate_closed_loop(ocp, ocp_solver, integrator, x0, N_sim=50,nMaxGuess: int = 1):
    
    nx = ocp.model.x.size()[0]  # Should be 14
    nu = ocp.model.u.size()[0]  # Should be 7

    # 初始状态
    simX = np.zeros((N_sim+1, nx))
    simU = np.zeros((N_sim, nu))
    simCost = np.zeros((N_sim, 1))
    simX[0, :] = x0  # 初始化状态为传入的 x0

    # 闭环仿真
    for i in range(N_sim):
        retries = 0
        success = True
        while retries < nMaxGuess:
            try:
                u_opt = ocp_solver.solve_for_x0(x0_bar = simX[i, :])
                #设置下一个sim的初始猜测
                u_guess, x_guess = get_guess_from_solver_result(ocp_solver, config.Horizon)
                clear_solver_state(ocp_solver, config.Horizon) #按道理不太需要
                for j in range(config.Horizon):
                    ocp_solver.set(j, "u", u_guess[:, j])
                    ocp_solver.set(j, "x", x_guess[:, j])
                ocp_solver.set(config.Horizon, "x", x_guess[:, -1])

                simU[i, :] = u_opt
                # print("u_opt", u_opt)
                # 更新状态
                x_next = integrator.simulate(x=simX[i, :], u=u_opt)
                simX[i+1, :] = x_next
                simCost[i,:] = ocp_solver.get_cost()
                # print("i:",i,"x:",x_next)
                break
            except Exception as e:
                success = False
                print(f"Error in MPC_solve: {str(e)}, change guess")
                print("current step:", i, ", retries=", retries)
                time.sleep(2)
            retries += 1
            if retries == nMaxGuess-1:
                print("set initial guess = 2pi")
                for j in range(0,config.Horizon,20):
                        ocp_solver.set(j, "x", np.array([0,0,0,0,0,0]))
                ocp_solver.set(0, "x", np.array([0,0,0,0,0,0]))
        if success == False:
            print("MPC solve failed after max retries")
            break
            

    # print("x_final:", simX[-1,:])
    clear_solver_state(ocp_solver, config.Horizon)

    t = np.linspace(0, N_sim*config.Ts, N_sim+1)
    return t, simX, simU, simCost, success

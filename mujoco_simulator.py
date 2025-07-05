
import numpy as np
import mujoco
import os
import config

class MuJoCoSimulator:

    
    def __init__(self, xml_path=None):

        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), 'xml', 'mjx_scene.xml')
        

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        

        self.n_joints = self.model.nv  # 关节数
        self.n_actuators = self.model.nu  # 执行器数
        
        print(f"MuJoCo模型加载成功:")
        print(f"  关节数: {self.n_joints}")
        print(f"  执行器数: {self.n_actuators}")
        print(f"  时间步长: {self.model.opt.timestep}")
        
        # 设置仿真参数
        self.dt = config.Ts  # 使用config中的时间步长
        self.substeps = max(1, int(self.dt / self.model.opt.timestep))
        
        print(f"  MPC时间步长: {self.dt}")
        print(f"  每个MPC步的子步数: {self.substeps}")
    
    def reset(self, q_init=None, qd_init=None):
        # 重置数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始关节位置和速度
        if q_init is not None:
            self.data.qpos[:len(q_init)] = q_init
        if qd_init is not None:
            self.data.qvel[:len(qd_init)] = qd_init
        
        mujoco.mj_forward(self.model, self.data)
    
    def step(self, u):
        # 设置控制输入
        self.data.ctrl[:len(u)] = u
        
        # 执行多个子步骤
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        
        q = self.data.qpos[:7].copy()
        qd = self.data.qvel[:7].copy()
        x_next = np.concatenate([q, qd])
        
        return x_next
    
    def get_state(self):
        q = self.data.qpos[:7].copy()
        qd = self.data.qvel[:7].copy()
        return np.concatenate([q, qd])
    
    def get_end_effector_pos(self):
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'panda_link8')
                return self.data.xpos[body_id].copy()
            except:
                
                print("Warning: 手动计算末端执行器位置")
                return self.data.xpos[-1].copy()
    
    def compare_end_effector_pos(self, T_fk_fun):

        # MuJoCo的末端位置
        mujoco_ee_pos = self.get_end_effector_pos()
        
        # urdf2casadi的末端位置
        q_current = self.data.qpos[:7]
        casadi_ee_pos = self.fk_position_casadi(T_fk_fun, q_current)
        
        print(f"MuJoCo末端位置: {mujoco_ee_pos}")
        print(f"CasADi末端位置: {casadi_ee_pos}")
        print(f"位置差异: {np.linalg.norm(mujoco_ee_pos - casadi_ee_pos):.4f}m")
        
        return mujoco_ee_pos, casadi_ee_pos
    
    def fk_position_casadi(self, T_fk_fun, q_row):
        T = T_fk_fun(q_row)       
        p = T[:3, 3]               
        return np.array(p).reshape(3)

def simulate_closed_loop_mujoco(ocp, ocp_solver, mujoco_sim, x0, N_sim=50, nMaxGuess: int = 1):

    
    nx = ocp.model.x.size()[0]  # Should be 14
    nu = ocp.model.u.size()[0]  # Should be 7

    # 初始状态
    simX = np.zeros((N_sim+1, nx))
    simU = np.zeros((N_sim, nu))
    simCost = np.zeros((N_sim, 1))
    simX[0, :] = x0  # 初始化状态为传入的 x0

    # 初始化MuJoCo仿真器
    q_init = x0[:7]
    qd_init = x0[7:]
    mujoco_sim.reset(q_init, qd_init)

    # 闭环仿真
    for i in range(N_sim):
        retries = 0
        success = True
        while retries < nMaxGuess:
            try:
                # 使用Acados求解器求解最优控制
                u_opt = ocp_solver.solve_for_x0(x0_bar = simX[i, :])
                
                # 设置下一个sim的初始猜测
                u_guess, x_guess = get_guess_from_solver_result(ocp_solver, config.Horizon)
                clear_solver_state(ocp_solver, config.Horizon)
                for j in range(config.Horizon):
                    ocp_solver.set(j, "u", u_guess[:, j])
                    ocp_solver.set(j, "x", x_guess[:, j])
                ocp_solver.set(config.Horizon, "x", x_guess[:, -1])

                simU[i, :] = u_opt
                
                # 使用MuJoCo仿真器更新状态
                x_next = mujoco_sim.step(u_opt)
                simX[i+1, :] = x_next
                simCost[i,:] = ocp_solver.get_cost()
                
                break
            except Exception as e:
                success = False
                print(f"Error in MPC_solve: {str(e)}, change guess")
                print("current step:", i, ", retries=", retries)
                import time
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

    # 清理求解器状态
    clear_solver_state(ocp_solver, config.Horizon)

    t = np.linspace(0, N_sim*config.Ts, N_sim+1)
    return t, simX, simU, simCost, success

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

def clear_solver_state(ocp_solver, N_horizon):
    for i in range(N_horizon):
        ocp_solver.set(i, "x", np.zeros_like(ocp_solver.get(i,"x")))
        ocp_solver.set(i, "u", np.zeros_like(ocp_solver.get(i,"u")))
    ocp_solver.set(N_horizon, "x", np.zeros_like(ocp_solver.get(N_horizon,"x")))

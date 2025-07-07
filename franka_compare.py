

import numpy as np
from acados_template import AcadosSim, AcadosSimSolver
import mujoco
import config
from franka_model import export_franka_ode_model
from mujoco_simulator import MuJoCoSimulator
from franka_closed_loop import create_ocp_solver
import pinocchio as pin

def rk4_aba_step(q, v, tau, h, model, data):
    """单个 RK4 子步（显式四阶）"""
    def acc(q_, v_):
        return pin.aba(model, data, q_, v_, tau)
    k1q = v
    k1v = acc(q,             v)
    k2q = v + 0.5*h*k1v
    k2v = acc(q + 0.5*h*k1q, v + 0.5*h*k1v)
    k3q = v + 0.5*h*k2v
    k3v = acc(q + 0.5*h*k2q, v + 0.5*h*k2v)
    k4q = v +     h*k3v
    k4v = acc(q +     h*k3q, v +     h*k3v)
    q_next = q + (h/6.0)*(k1q + 2*k2q + 2*k3q + k4q)
    v_next = v + (h/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    return q_next, v_next

def pin_integrate_rk4(q0, v0, tau, dt, n_sub):
    """与 acados 对齐：dt=0.01, n_sub=10, h=0.001"""
    h = dt / n_sub
    model = pin.buildModelFromUrdf('panda_arm.urdf')
    data  = model.createData()
    q, v = q0.copy(), v0.copy()
    for _ in range(n_sub):
        q, v = rk4_aba_step(q, v, tau, h, model, data)
    return np.concatenate([q, v])

def create_sim_integrator():
    """生成 tf = config.Ts 的 SimSolver，用于与 MuJoCo 对比"""
    sim = AcadosSim()
    sim.model = export_franka_ode_model()

    # 显式四阶 Runge–Kutta，与 MuJoCo 的 RK4 对齐
    sim.solver_options.T = 0.01  # 0.01 s
    sim.solver_options.integrator_type = 'ERK'
    sim.solver_options.sim_method_num_stages = 4
    sim.solver_options.sim_method_num_steps = 1
    acados_integrator = AcadosSimSolver(sim)

 
    return acados_integrator


def compare_one_step():
    """单步仿真对比"""
    x0 = np.zeros(14)
    # 任意关节力矩输入（注意逗号）
    u = np.array([1.0, 1.0, 1.0, 3.8, -1.54, 1.0, 1.89])

    # 仅为初始化 ocp_solver，可不直接使用
    sim = create_sim_integrator()

    mujoco = MuJoCoSimulator()
    # acados
    sim.set('x', x0)
    sim.set('u', u)
    sim.solve()
    x_next_acados = sim.get('x')

    # MuJoCo
    mujoco.reset(x0[:7], x0[7:])
    x_next_mujoco = mujoco.step(u)
    x_next_pin = pin_integrate_rk4(x0[:7], x0[7:], u, 0.01, 10)
    print('=== 单步仿真对比 ===')
    print('acados :', x_next_acados)
    print('mujoco :', x_next_mujoco)
    print('diff   :', np.abs(x_next_acados - x_next_mujoco))
    print("\n=== acados vs Pinocchio-integrator ===")
    print("pinocchio-integrator:", x_next_pin)
    print("max |diff| =", np.max(np.abs(x_next_acados - x_next_pin)))
    print("component-wise diff =", x_next_acados - x_next_pin)


def compare_n_steps(n=10):
    """连续 n 步仿真对比"""
    x0 = np.zeros(14)
    u = np.zeros(7)

    sim = create_sim_integrator()

    mujoco = MuJoCoSimulator()

    x_sim = x0.copy()
    x_mj = x0.copy()

    print(f'=== 连续 {n} 步仿真对比 ===')
    for i in range(n):
        # acados
        sim.set('x', x_sim)
        sim.set('u', u)
        sim.solve()
        x_sim = sim.get('x')

        # MuJoCo
        if i == 0:
            mujoco.reset(x_mj[:7],x_mj[7:])
        x_mj = mujoco.step(u)

        print(f'Step {i+1:02d}: max |diff| = {np.max(np.abs(x_sim - x_mj))}')


def main():
    print("MuJoCo version:", mujoco.__version__)
    
    model = mujoco.MjModel.from_xml_path("panda_arm.xml")
    data  = mujoco.MjData(model)
    print("opt.integrator:", model.opt.integrator)
    q = np.zeros(model.nq)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    import pinocchio as pin
    urdf = "panda_arm.urdf"
    model = pin.buildModelFromUrdf(urdf)
    data  = model.createData()
    q = pin.neutral(model)               # 同样置零
    M_pin = pin.crba(model, data, q)
    print("Mujoco M:", M)
    print("Pinocchio M:", M_pin)
    print("Mujoco M - Pinocchio M:", np.abs(M - M_pin))
    if np.allclose(M, M_pin):
        print("Mujoco 和 Pinocchio 的质量矩阵一致！")
    # 这里停止10秒，观察输出
    import time
    time.sleep(10)
    compare_one_step()
    compare_n_steps(10)


if __name__ == '__main__':
    main()

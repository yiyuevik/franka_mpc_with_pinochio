import mujoco, pinocchio as pin, numpy as np
from acados_template import AcadosSim, AcadosSimSolver
from franka_model import export_franka_ode_model


q0   = np.zeros(7)
v0   = np.zeros(7)
tau  = np.array([ 1, 1, 1, 3.8, -1.54, 1, 1.89])

# MuJoCo ABA
m   = mujoco.MjModel.from_xml_path('panda_arm.xml')
d   = mujoco.MjData(m)
d.qpos[:]          = q0
d.qvel[:]          = v0
d.ctrl[:] = tau
mujoco.mj_forward(m, d)
qdd_mj = d.qacc.copy()

# Pinocchio ABA
pin_model = pin.buildModelFromUrdf('panda_arm.urdf')
pin_data  = pin_model.createData()
qdd_pin   = pin.aba(pin_model, pin_data, q0, v0, tau)

# acados
sim = AcadosSim()
sim.model = export_franka_ode_model()
f_casadi     = sim.model.f_expl_expr      # CasADi SX expression
from casadi import Function
f_fun = Function('f', [sim.model.x, sim.model.u, sim.model.p], [f_casadi])

sim.solver_options.integrator_type        = 'ERK'
sim.solver_options.sim_method_num_steps   = 1
sim.solver_options.sim_method_num_stages  = 4


x_sym = np.concatenate([q0, v0])
xdot  = f_fun(x_sym, tau, np.zeros(0))      
qdd_ac = np.array(xdot).flatten()[7:]      # xdot = [q̇; q̈]

# 对比
print('max |acados - mj | =', np.max(np.abs(qdd_pin - qdd_mj)))
print('qdd_mj:', qdd_mj)
print('qdd_pin:', qdd_pin)
print('qdd_ac:', qdd_ac)
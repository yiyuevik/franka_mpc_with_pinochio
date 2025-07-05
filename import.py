import casadi as ca
import numpy as np
import urdf2casadi.urdfparser as u2c

# 加载模型
parser = u2c.URDFparser()
parser.from_file("urdf/panda_arm.urdf")
fk = parser.get_forward_kinematics("panda_link0", "panda_link8")
T_fk_fun = fk["T_fk"]

# 设定符号变量
q = ca.SX.sym("q", 7)
T_fk = T_fk_fun(q)
p_fk = T_fk[:3, 3]

# 目标位置
p_target = np.array([0.3, 0.3, 0.5])
loss = ca.sumsqr(p_fk - p_target)

# 构建优化器
opti = ca.Opti()
q_var = opti.variable(7)
opti.minimize(ca.sumsqr(T_fk_fun(q_var)[:3, 3] - p_target))
opti.subject_to(opti.bounded(-2*np.pi, q_var, 2*np.pi))
opti.solver("ipopt")

# 求解
sol = opti.solve()
q_sol = sol.value(q_var)
print("IK 解:", q_sol)

q_var = np.array([0.85114759,  0.22895749, -0.00465471, -1.69075889, -0.24523397,  0.20150836,
  0.     ])
print(T_fk_fun(q_var)[:3, 3])
import pinocchio as pin
from pinocchio import casadi as cpin     
import casadi as ca

# 1. 读取 URDF，构建普通 double 精度模型
urdf_path = "panda_arm.urdf"
model = pin.buildModelFromUrdf(urdf_path)

# 2. 转成 CasADi 标量模型，并创建配套 data
cmodel = cpin.Model(model)                   # SX 标量模型
cdata  = cmodel.createData()                 # 必须配套 CasADi Data

# 3. 定义符号列向量 (n×1)
q   = ca.SX.sym("q",   cmodel.nq, 1)         # 关节位姿
v   = ca.SX.sym("v",   cmodel.nv, 1)         # 关节速度
tau = ca.SX.sym("tau", cmodel.nv, 1)         # 关节力矩

# 4. 调动力学算法 —— 一定要传 cdata
M   = cpin.crba(cmodel, cdata, q)            # 关节惯性矩阵 M(q)

vdot = cpin.aba(cmodel, cdata, q, v, tau)    # Forward dynamics  \ddot q
xdot = ca.vertcat(v, vdot)                   # 状态微分 [ q̇ ; q̈ ]

# 5. 包装为 CasADi 函数，供 OCP / MPC 调用
f_dyn = ca.Function("f_dyn", [q, v, tau], [xdot])

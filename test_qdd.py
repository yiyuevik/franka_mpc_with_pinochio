
import numpy as np, mujoco, pinocchio as pin, os

def load_models():
    mj_model = mujoco.MjModel.from_xml_path('panda_arm.xml')
    mj_data  = mujoco.MjData(mj_model)
    pin_model = pin.buildModelFromUrdf('panda_arm.urdf')
    pin_data  = pin_model.createData()
    return mj_model, mj_data, pin_model, pin_data

def aba_pin(pin_model, pin_data, q, qd, tau):
    return pin.aba(pin_model, pin_data, q, qd, tau)

def aba_mj(mj_model, mj_data, q, qd, tau):
    mj_data.qpos[:] = q
    mj_data.qvel[:] = qd
    mujoco.mj_forward(mj_model, mj_data)         # 得到 qfrc_bias 和 M
    M = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M, mj_data.qM)
    qdd = np.linalg.solve(M, tau - mj_data.qfrc_bias)
    return qdd

if __name__ == "__main__":
    mj_model, mj_data, pin_model, pin_data = load_models()
    q   = np.zeros(mj_model.nq)     # 关节角置零
    qd  = np.zeros(mj_model.nv)     # 关节速置零

    print("=== 逐关节 +1 N·m → qdd 对比 ===")
    for j in range(mj_model.nv):
        tau = np.zeros(7); tau[j] = 1.0
        qdd_mj  = aba_mj(mj_model, mj_data, q, qd, tau)
        qdd_pin = aba_pin(pin_model, pin_data, q, qd, tau)
        sign_ok = np.sign(qdd_mj[j]) == np.sign(qdd_pin[j])
        print(f"joint{j+1}:  MuJoCo={qdd_mj[j]: .3e}, Pin={qdd_pin[j]: .3e},  same sign? {sign_ok}")

"""
cartpole_utils.py

我先放了一些辅助函数：如绘制状态/控制量曲线，以及简单动画等
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi as ca, os

def build_fk():
    urdf = os.path.join(os.path.dirname(__file__),  "panda_arm.urdf")
    model  = pin.buildModelFromUrdf(urdf)
    cmodel = cpin.Model(model)
    cdata  = cmodel.createData()
    q_sym  = ca.SX.sym("q", 7, 1)
    cpin.framesForwardKinematics(cmodel, cdata, q_sym)
    p_ee   = cdata.oMf[cmodel.getFrameId("panda_link8")].translation
    return ca.Function("fk_pos", [q_sym], [p_ee])

def plot_cartpole_trajectories(t, simX, simU=None):
    fig, axs = plt.subplots(7, 1, figsize=(6,10), sharex=True)

    labels = ['x', 'xdot', 'theta1', 'omega1', 'theta2', 'omega2']
    for i in range(6):
        axs[i].plot(t, simX[:, i], label=labels[i])
        axs[i].grid(True)
        axs[i].legend()

    if simU is not None:
        axs[5].plot(t[:-1], simU[:,0], label='Force')
        axs[5].grid(True)
        axs[5].legend()

    axs[-1].set_xlabel("time (s)")
    plt.suptitle("Double Pendulum Closed-Loop Trajectories")

    plt.tight_layout()
    plt.show()


def animate_cartpole(t, X, L1=1.0, L2=1.0, interval=50):
    """
    Animates the CartPole system.

    Args:
    - t: Time vector.
    - X: State matrix (each row is a state at a time step, with columns for x, x_dot, theta, and theta_dot).
    - L: Length of the pendulum (default is 1.0).
    - interval: Time interval between frames in milliseconds (default is 50).
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim([-10, 10])  # x-axis limits for cart movement
    ax.set_ylim([-10, 5])  # y-axis limits for cart and pendulum movement
    ax.set_aspect('equal')
    ax.set_title('Double Pendulum Animation')

    # Cart dimensions
    cart_width = 0.5
    cart_height = 0.3

    # Create cart (black rectangle)
    cart = plt.Rectangle((0, 0), cart_width, cart_height, color='black', animated=True)
    ax.add_patch(cart)

    # 创建两根摆杆的线条
    line1, = ax.plot([], [], lw=2, color='blue')  # 第一根摆杆
    line2, = ax.plot([], [], lw=2, color='red')  # 第二根摆杆

    # Update function for animation
    def update(frame):
        # Get the current state from X
        x_cart = X[frame, 0]
        theta1 = X[frame, 2]
        theta2 = X[frame, 4]

        # Update the cart position
        cart.set_xy((x_cart - cart_width/2, 0))

        # Calculate the position of the pole
        pole_top_x = x_cart
        pole_top_y = cart_height
        x1 = pole_top_x + L1 * np.sin(theta1)
        y1 = pole_top_y + L1 * np.cos(theta1)

        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 + L2 * np.cos(theta2)

        # Update the ple line
        line1.set_data([pole_top_x, x1], [pole_top_y, y1])
        line2.set_data([x1, x2], [y1, y2])

        return cart, line1, line2

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=interval)
    plt.show()


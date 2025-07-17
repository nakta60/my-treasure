import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 시스템 정의
def system(t, state, r, K, beta, theta_1, theta_2, a, alpha_1, alpha_2, alpha_3, mu_1, mu_2, e_1, e_2, e_3, d_1, d_2, lamda_1, lamda_2):
    A_h, A_i, Z, P = state
    A_h = np.clip(A_h, 0, 1e15)
    A_i = np.clip(A_i, 0, 1e15)
    Z = np.clip(Z, 0, 1e15)
    P = np.clip(P, 0, 1e15)
    T = (theta_1 * A_h + theta_2 * A_i)
    beta_T = beta * T / (1 + a * T)
    g_T = T / (1 + a * T)

    dA_h = r * A_h * (1 - A_h/K) - beta_T * A_h * P - alpha_1 * g_T * A_h * Z
    dA_i = beta_T * A_h * P - mu_1 * A_i - alpha_2 * g_T * A_i * Z
    dZ = e_1 * alpha_1 * g_T * A_h * Z + e_2 * alpha_2 * g_T * A_i * Z + e_3 * alpha_3 * Z * P - d_1 * Z - d_2 * Z**2
    dP = lamda_1 * mu_1 * A_i + lamda_2 * alpha_2 * g_T * A_i * Z - mu_2 * P - alpha_3 * P * Z - beta_T * A_h * P

    return np.array([dA_h, dA_i, dZ, dP])

# RK4 스텝
def rk4_step(f, t, y, h, *args):
    k1 = h * f(t, y, *args)
    k2 = h * f(t + h / 2, y + k1 / 2, *args)
    k3 = h * f(t + h / 2, y + k2 / 2, *args)
    k4 = h * f(t + h, y + k3, *args)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# 파라미터 설정
r = 1.5
K = 10000
beta = 1
theta_1 = 1
theta_2 = 3
a = 5
alpha_1 = 0.5
alpha_2 = 0.7
alpha_3 = 0.2
mu_1 = 0.3
mu_2 = 0.2
e_1 = 0.5
e_2 = 0.4
e_3 = 0.3
d_1 = 0.1
d_2 = 0.05
lamda_1 = 10
lamda_2 = 5 

y0 = np.array([5000, 0, 1000 ,100])
t0 = 0
t_end = 10000
h = 0.001

num_steps = int((t_end - t0) / h)

# 결과 저장용 배열
t_values = np.linspace(t0, t_end, num_steps)

states = y0

# RK4 루프 실행(Pre_iteration)
for i in range(1, num_steps):
    states = rk4_step(system, t_values[i-1], states, h, r, K, beta, theta_1, theta_2, a, alpha_1, alpha_2, alpha_3, mu_1, mu_2, e_1, e_2, e_3, d_1, d_2, lamda_1, lamda_2)


# RK4 루프 실행
t0_ = 0
t_end_ = 100
h_ = 0.001

num_steps_ = int((t_end-t0_) / h_)

t_values_ = np.linspace(t0_, t_end_, num_steps_)
states_ = np.zeros((num_steps_, 4))
states_[0] = states

for i in range(1, num_steps_):
    states_[i] = rk4_step(system, t_values[i-1], states_[i-1], h, r, K, beta, theta_1, theta_2, a, alpha_1, alpha_2, alpha_3, mu_1, mu_2, e_1, e_2, e_3, d_1, d_2, lamda_1, lamda_2)


# 3D 플롯 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 선 플롯 그리기
ax.plot(states_[:,0], states_[:,2], states_[:,3], color = 'red')
ax.legend()

# 시작점 끝점 표시
ax.scatter([states_[0,0]],[states_[0,2]],[states_[0,3]], color = 'lime')
ax.scatter([states_[-1,0]],[states_[-1,2]],[states_[-1,3]], color = 'grey')

# 축 라벨 설정
ax.set_xlabel('Algae')
ax.set_ylabel('Zooplankton')
ax.set_zlabel('Parasite')


# 플롯 표시

plt.show()

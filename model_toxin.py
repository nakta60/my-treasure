import numpy as np
import matplotlib.pyplot as plt

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

# 초기 조건 및 시간 설정
y0 = np.array([5000, 0, 1000, 100])
t0 = 0
t_end = 100
h = 0.001
num_steps = int((t_end - t0) / h)
t_values = np.linspace(t0, t_end, num_steps)
states = np.zeros((num_steps, 4))
states[0] = y0

# 시뮬레이션 실행
for i in range(1, num_steps):
    states[i] = rk4_step(system, t_values[i - 1], states[i - 1], h,
                         r, K, beta, theta_1, theta_2, a, alpha_1, alpha_2, alpha_3,
                         mu_1, mu_2, e_1, e_2, e_3, d_1, d_2, lamda_1, lamda_2)

# 시각화
fig, ax1 = plt.subplots(figsize=(10, 6))

line1, = ax1.plot(t_values, states[:, 1], color='skyblue', label='infected algae')
line2, = ax1.plot(t_values, states[:, 2], color='blue', label='zooplankton')
line3, = ax1.plot(t_values, states[:, 3], color='royalblue', label='parasite')
ax1.set_xlabel('Time')
ax1.set_ylabel('Population size (A_i, Z, P)')
ax1.grid(True)

ax2 = ax1.twinx()
line4, = ax2.plot(t_values, states[:, 0], color='magenta', label='healthy algae')
ax2.set_ylabel('Healthy algae size (A_h)')

# 통합 범례 추가
lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Algae-Parasite-Zooplankton System with Toxin-Mediated Dynamics")
plt.tight_layout()
plt.show()

print(states[-10:,0])

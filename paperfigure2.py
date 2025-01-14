import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 시스템 방정식 정의
def system(t, vars):
    x_I, x_U, y_I, y_U, z = vars

    # 논문에서 제시된 매개변수
    K = 2000       # Carrying capacity
    d_x = 0.1      # Death rate of prey
    d_y = 1      # Death rate of predator
    d_z = 0.09      # Death rate of parasite
    Q_x = 0.2      # Fraction of infected prey contributing to parasite growth
    Q_y = 0.8      # Fraction of infected predator reproduction
    f_y = 0.01      # Predation rate
    S = 0.0005        # Parasite growth rate
    g_x = 2      # Growth rate of uninfected prey
    r_x = 1      # Conversion rate of infected prey
    k_y = 0.2      # Predator growth rate
    r_p = 1      # Reproductive success of infected predator
    r_e = 1      # Reproductive success of uninfected predator
    n_z = 6      # Parasite reproduction rate

    # 미분 방정식 정의
    dx_I = -((x_I + x_U) * x_I) / K - d_x * x_I + Q_x * S * x_U * z - f_y * x_I * (y_I + y_U)
    dx_U = g_x * (r_x * x_I + x_U) - ((x_I + x_U) * x_U) / K - d_x * x_U - Q_x * S * x_U * z - f_y * x_U * (y_I + y_U)
    dy_I = -d_y * y_I + Q_y * f_y * x_I * y_U
    dy_U = (k_y * f_y * x_U * (r_p * y_I + y_U) +
            (r_e * k_y * (1 - Q_y) - (1 - r_p * k_y) * Q_y) * f_y * x_I * y_U +
            (r_p * r_e * k_y * (1 - Q_y) + r_p**2 * k_y * Q_y) * f_y * x_I * y_I -
            d_y * y_U)
    dz = -Q_x * S * x_U * z + n_z * Q_y * f_y * x_I * (y_I + y_U) - d_z * z

    return [dx_I, dx_U, dy_I, dy_U, dz]

# 초기값 설정
initial_conditions = [50, 50, 10, 10, 10]  # 초기 상태: [x_I, x_U, y_I, y_U, z]
t_span = (0, 150)  # 시간 범위
time_points = np.linspace(t_span[0], t_span[1], 1000)  # 시간 샘플링

# solve_ivp 실행
solution = solve_ivp(system, t_span, initial_conditions, method='RK45', t_eval=time_points)

# 결과 시각화
fig, ax1 = plt.subplots()

# 왼쪽 y축: 피식자와 포식자
ax1.plot(solution.t, solution.y[0], label='x_I (Infected Prey)', color='gray', linestyle='--')
ax1.plot(solution.t, solution.y[1], label='x_U (Uninfected Prey)', color='cyan', linestyle='-.')
ax1.plot(solution.t, solution.y[2], label='y_I (Infected Predator)', color='black', linestyle='-')
ax1.plot(solution.t, solution.y[3], label='y_U (Uninfected Predator)', color='green', linestyle=':')
ax1.set_xlabel('Time')
ax1.set_ylabel('Host Population Size')
ax1.legend(loc='upper left')
ax1.grid()

# 오른쪽 y축: 기생충
ax2 = ax1.twinx()
ax2.plot(solution.t, solution.y[4], label='Parasite', color='magenta', linestyle='-')
ax2.set_ylabel('Parasite Population Size', color='magenta')
ax2.tick_params(axis='y', labelcolor='magenta')

# 제목 추가
fig.suptitle('Host-Parasite Dynamics with Parasite on Secondary Axis')

# 범례 추가 (오른쪽 축)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

plt.tight_layout()
plt.show()

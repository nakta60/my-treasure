import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 동족포식 확률 기록을 위한 리스트
dynamic_c_y = []

# 시스템 방정식 정의 (동족포식 확률이 동적으로 변함)
def system(t, vars):
    x_I, x_U, y_I, y_U, z = vars

    # 매개변수 설정
    K = 2000
    d_x = 0.1
    d_y = 1
    d_z = 0.09
    Q_x = 0.1
    Q_y = 0.6
    f_y = 0.01
    S = 0.0005
    g_x = 2
    r_x = 1
    k_y = 0.2
    r_p = 1
    r_e = 1
    n_z = 6

    # 동족포식 확률 계산
    T_H = 500  # 피식자 밀도 임계값
    c0 = 0.15  # 기본 동족포식 확률
    alpha = 0.4  # 동족포식 확률 변화폭
    k = 0.1  # 동족포식 변화 속도
    total_prey = x_I + x_U

    # 새로운 동족포식 확률 함수
    c_y = c0 + alpha * np.tanh(k * (T_H - total_prey))
    c_y = np.clip(c_y, 0, 1)  # 동족포식 확률 제한

    # 동족포식 확률 기록
    dynamic_c_y.append(c_y)

    # 값 제한 (오버플로우 방지)
    x_I = np.clip(x_I, 0, 1e6)
    x_U = np.clip(x_U, 0, 1e6)
    y_I = np.clip(y_I, 0, 1e6)
    y_U = np.clip(y_U, 0, 1e6)
    z = np.clip(z, 0, 1e6)

    # 미분 방정식 정의
    dx_I = -((x_I + x_U) * x_I) / K - d_x * x_I + Q_x * S * x_U * z - f_y * x_I * (y_I + y_U)
    dx_U = g_x * (r_x * x_I + x_U) - (x_U * x_U) / K - d_x * x_U - Q_x * S * x_U * z - f_y * x_U * (y_I + y_U)
    dy_I = -d_y * y_I + Q_y * f_y * x_I * y_U - c_y * y_I * y_U + c_y * Q_y * y_I * y_U 
    dy_U = k_y * f_y * x_U * (r_p * y_I + y_U) + (r_e * k_y * (1 - Q_y) - (1 - r_p * k_y) * Q_y) * f_y * x_I * y_U - d_y * y_U + (r_p * r_e * k_y * (1 - Q_y) + r_p ** 2 * k_y * Q_y) * f_y * x_I * y_I - c_y * Q_y * y_I * y_U + c_y * Q_y * k_y * r_p * y_I * y_U + r_e * (1 - Q_y) * c_y * k_y * y_I * y_U
    dz = -Q_x * S * x_U * z + n_z * Q_y * f_y * x_I * (y_I + y_U) - d_z * z + n_z * Q_y * c_y * y_I * y_U

    return [dx_I, dx_U, dy_I, dy_U, dz]

# 초기값 설정
initial_conditions = [0, 800, 0, 100, 1000]
t_span = (0, 300)
time_points = np.linspace(t_span[0], t_span[1], 30000)

# solve_ivp 실행
solution = solve_ivp(system, t_span, initial_conditions, method='BDF', t_eval=time_points, rtol=1e-3, atol=1e-6)


# 결과 확인
print("Integration successful:", solution.success)
print("Message:", solution.message)

# 서브플롯 생성
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))

# 첫 번째 서브플롯: 동족포식 확률

ax1.plot(dynamic_c_y, label='Cannibalism Probability (c_y)', color='red')
ax1.set_xlabel('Time')
ax1.set_ylabel('Cannibalism Probability')
ax1.set_title('Dynamic Cannibalism Probability Over Time')
ax1.legend()
ax1.grid()

# 두 번째 서브플롯: 인구 변화
# 왼쪽 y축: 피식자와 포식자
ax3 = ax2
ax3.plot(solution.t, solution.y[0], label='x_I (Infected Prey)', color='gray', linestyle='-')
ax3.plot(solution.t, solution.y[1], label='x_U (Uninfected Prey)', color='cyan', linestyle='-')
ax3.plot(solution.t, solution.y[2], label='y_I (Infected Predator)', color='black', linestyle='-')
ax3.plot(solution.t, solution.y[3], label='y_U (Uninfected Predator)', color='green', linestyle='-')
ax3.set_xlabel('Time')
ax3.set_ylabel('Host Population Size')
ax3.grid()

# 오른쪽 y축: 기생충
ax4 = ax3.twinx()
ax4.plot(solution.t, solution.y[4], label='z (Parasite)', color='magenta', linestyle='-')
ax4.set_ylabel('Parasite Population Size', color='magenta')
ax4.tick_params(axis='y', labelcolor='magenta')

# 범례 설정
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# 제목 및 레이아웃
ax3.set_title('Population Dynamics Over Time')
plt.tight_layout()
plt.show()

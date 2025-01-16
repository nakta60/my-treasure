
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 시스템 방정식 정의
def system_dynamic(t, vars, c_y):
    x_I, x_U, y_I, y_U, z = vars

    # 매개변수 설정
    K = 2000
    d_x = 0.1
    d_y = 1
    d_z = 0.09
    Q_x = 0.3
    Q_y = 0.4
    f_y = 0.01
    S = 0.0005
    g_x = 2
    r_x = 1
    k_y = 0.2
    r_p = 1
    r_e = 1
    n_z = 6

    # 미분 방정식 정의
    dx_I = -((x_I + x_U) * x_I) / K - d_x * x_I + Q_x * S * x_U * z - f_y * x_I * (y_I + y_U)
    dx_U = g_x * (r_x * x_I + x_U) - ((x_I + x_U) * x_U) / K - d_x * x_U - Q_x * S * x_U * z - f_y * x_U * (y_I + y_U)
    dy_I = -d_y * y_I + Q_y * f_y * x_I * y_U - c_y * y_I * y_U + Q_y * c_y * y_I * y_U - c_y * y_I**2
    dy_U = k_y * f_y * x_U * (r_p * y_I + y_U) + (r_e * k_y * (1 - Q_y) - (1 - r_p * k_y) * Q_y) * f_y * x_I * y_U - d_y * y_U - c_y * y_U**2
    dz = -Q_x * S * x_U * z + n_z * Q_y * f_y * x_I * (y_I + y_U) - d_z * z + (Q_y * c_y * y_I * y_U + Q_y * c_y * y_I**2) * n_z

    return [dx_I, dx_U, dy_I, dy_U, dz]

# c_y 값에 따른 분기 다이어그램 계산
c_y_values = np.linspace(0, 1.0, 1000)  # c_y 범위
results = []  # 안정 상태 결과 저장

for c_y in c_y_values:
    # 초기값 및 시간 설정
    initial_conditions = [0, 3000, 0, 100, 1000]
    t_span = (0, 150)
    time_points = np.linspace(t_span[0], t_span[1], 1000)

    # 적분 수행
    solution = solve_ivp(system_dynamic, t_span, initial_conditions, args=(c_y,), method='BDF', t_eval=time_points, rtol=1e-3, atol=1e-6)

    # 안정 상태 추출 (마지막 시간 단계의 값)
    if solution.success:
        steady_state = solution.y[:, -1]  # 마지막 시점의 값
        results.append((c_y, steady_state))
    else:
        results.append((c_y, [np.nan] * 5))  # 적분 실패 시 NaN 저장

# 결과 분리
c_y_values, steady_states = zip(*results)
steady_states = np.array(steady_states)

# 분기 다이어그램 플롯
plt.figure(figsize=(10, 6))

# 감염된 피식자 (x_I) 안정 상태 플롯
plt.plot(c_y_values, steady_states[:, 0], color = 'gray', label='x_I (Infected Prey)')

# 비감염된 피식자 (x_U) 안정 상태 플롯
plt.plot(c_y_values, steady_states[:, 1], color = 'cyan', label='x_U (Uninfected Prey)')

# 감염된 포식자 (y_I) 안정 상태 플롯
plt.plot(c_y_values, steady_states[:, 2], color = 'black', label='y_I (Infected Predator)')

# 비감염된 포식자 (y_U) 안정 상태 플롯
plt.plot(c_y_values, steady_states[:, 3], color = 'green', label='y_U (Uninfected Predator)')

# 기생충 (z) 안정 상태 플롯
plt.plot(c_y_values, steady_states[:, 4], color = 'magenta', label='z (Parasite)')

# 플롯 설정
plt.xlabel('c_y (Interaction Coefficient)')
plt.ylabel('Steady State Values')

plt.legend()
plt.grid()
plt.show()

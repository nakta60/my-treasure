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
    Q_x = 1
    Q_y = 1
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
    dx_U = g_x * (r_x * x_I + x_U) - (x_U * x_U) / K - d_x * x_U - Q_x * S * x_U * z - f_y * x_U * (y_I + y_U)
    dy_I = -d_y * y_I + Q_y * f_y * x_I * y_U - c_y*y_I*y_U + Q_y*c_y*y_I*y_U
    dy_U = k_y * f_y * x_U * (r_p * y_I + y_U) + (r_e * k_y * (1 - Q_y) - (1 - r_p * k_y) * Q_y) * f_y * x_I * y_U - d_y * y_U + (r_p * r_e * k_y * (1 - Q_y) + r_p ** 2 * k_y * Q_y) * f_y * x_I * y_I - c_y * Q_y * y_I * y_U + c_y * k_y * r_p * y_I * y_U * Q_y + r_e * (1 - Q_y) * c_y
    dz = -Q_x * S * x_U * z + n_z * Q_y * f_y * x_I * (y_I + y_U) - d_z * z + n_z * Q_y * c_y * y_I * y_U
    return [dx_I, dx_U, dy_I, dy_U, dz]

# c_y 값에 따른 분기 다이어그램 계산
c_y_values = np.linspace(0.1, 1.0, 100)  # c_y 범위

plt.figure(figsize=(10, 6))

for c_y in c_y_values:
    # 초기값 및 시간 설정
    initial_conditions = [0, 800, 0, 100, 1000]
    t_span = (0, 150)
    time_points = np.linspace(t_span[0], t_span[1], 1000)

    # 적분 수행
    solution = solve_ivp(system_dynamic, t_span, initial_conditions, args=(c_y,), method='BDF', t_eval=time_points, rtol=1e-3, atol=1e-6)
    
    if solution.success:
        steady_states = solution.y[:, -100:]  # 마지막 100개의 값
        c_y_array = np.full(steady_states.shape[1], c_y)  # c_y 값을 복제하여 x축 배열 생성
        
        # 각 상태 변수에 대해 플롯
        plt.plot(c_y_array, steady_states[0, :], 'b.', alpha=0.5, label='x_I (Infected Prey)' if c_y == c_y_values[0] else "")
        plt.plot(c_y_array, steady_states[1, :], 'c.', alpha=0.5, label='x_U (Uninfected Prey)' if c_y == c_y_values[0] else "")
        plt.plot(c_y_array, steady_states[2, :], 'r.', alpha=0.5, label='y_I (Infected Predator)' if c_y == c_y_values[0] else "")
        plt.plot(c_y_array, steady_states[3, :], 'g.', alpha=0.5, label='y_U (Uninfected Predator)' if c_y == c_y_values[0] else "")
        plt.plot(c_y_array, steady_states[4, :], 'm.', alpha=0.5, label='z (Parasite)' if c_y == c_y_values[0] else "")

# 플롯 설정
plt.xlabel('c_y')
plt.ylabel('State Variables')
plt.title('Plot last 100value')
plt.legend()
plt.grid()
plt.show()

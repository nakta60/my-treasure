import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors

# 동족포식 확률 기록을 위한 리스트
dynamic_c_y = []

# 시스템 방정식 정의
def system_dynamic(t, vars, r_p, r_x):
    x_I, x_U, y_I, y_U, z = vars

    # 매개변수 설정
    K = 2000
    d_x = 0.1
    d_y = 1
    d_z = 0.09
    Q_x = 0.88
    Q_y = 0.35
    f_y = 0.01
    S = 0.0005
    g_x = 2
    
    k_y = 0.2
    
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
    dynamic_c_y.append((t, c_y))

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

# 상태 분류 함수
def classify_state(solution):
    x_I, x_U, y_I, y_U, z = solution.y[:, -1]  # 마지막 시간 단계 값
    if x_U >= 1 and y_U >= 1 and z >= 1 and x_U >= x_I and y_U >= y_I:  
        return 1  # Magenta
    elif x_U >= 1 and y_U >= 1 and z >= 1 and x_U >= x_I and y_U < y_I:  
        return 2  # Yellow
    elif x_U >= 1 and y_U >= 1 and z >= 1 and x_U < x_I and y_U < y_I:  
        return 3  # Cyan
    elif x_U >= 1 and y_U >= 1 and z >= 1 and x_U < x_I and y_U >= y_I:
        return 4  # Black
    else:
        return 0

# 파라미터 범위 설정
r_p_range = np.linspace(0, 1, 100)
r_x_range = np.linspace(0, 1, 100)
r_p, r_x = np.meshgrid(r_p_range, r_x_range)

# 결과 저장 배열 생성
results = np.zeros(r_p.shape)

# 파라미터 조합에 따라 시뮬레이션 수행
for i in range(r_p.shape[0]):
    for j in range(r_p.shape[1]):
        # 초기 조건 및 시간 설정
        initial_conditions = [0, 800, 0, 100, 1000]
        t_span = (0, 300)
        time_points = np.linspace(t_span[0], t_span[1], 30000)

        # 적분 수행
        solution = solve_ivp(system_dynamic, t_span, initial_conditions, args=(r_p[i, j], r_x[i, j]), method='BDF', t_eval=time_points)

        # 상태 분류 및 결과 저장
        if solution.success:
            results[i, j] = classify_state(solution)
        else:
            results[i, j] = 0  # 적분 실패 시 멸종으로 간주

# 색상 지정
cmap = mcolors.ListedColormap(['white', 'magenta', 'yellow', 'cyan', 'black'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# pcolormesh를 사용해 결과 시각화
plt.figure(figsize=(8, 6))
plt.pcolormesh(r_p, r_x, results, cmap=cmap, norm=norm, shading='auto')

plt.xlabel('$r_p$')
plt.ylabel('$r_x$')
plt.show()

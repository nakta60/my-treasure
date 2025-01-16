import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors

# 시스템 방정식 정의
def system_dynamic(t, vars, Q_y, r_p, Q_x):
    x_I, x_U, y_I, y_U, z = vars

    # 매개변수 설정
    K = 2000
    d_x = 0.1
    d_y = 1
    d_z = 0.09
    f_y = 0.01
    S = 0.0005
    g_x = 2
    r_x = 1
    r_e = 1
    k_y = 0.2
    n_z = 6
    c_y = 0.2
    # 미분 방정식 정의
    dx_I = -((x_I + x_U) * x_I) / K - d_x * x_I + Q_x * S * x_U * z - f_y * x_I * (y_I + y_U)
    dx_U = g_x * (r_x * x_I + x_U) - ((x_I + x_U) * x_U) / K - d_x * x_U - Q_x * S * x_U * z - f_y * x_U * (y_I + y_U)
    dy_I = -d_y * y_I + Q_y * f_y * x_I * y_U - c_y*y_I*y_U + Q_y*c_y*y_I*y_U - c_y*y_I**2
    dy_U = k_y * f_y * x_U * (r_p * y_I + y_U) + (r_e * k_y * (1 - Q_y) - (1 - r_p * k_y) * Q_y) * f_y * x_I * y_U - d_y * y_U -c_y*y_U**2+k_y*c_y*y_U**2+(r_p**2*k_y*Q_y+r_p*r_e*k_y*(1-Q_y))+c_y*y_I*y_U*(-Q_y+r_p*k_y*Q_y+r_e*k_y*(1-Q_y))+c_y*y_U*y_I*(-1+r_p*k_y)
    dz = -Q_x * S * x_U * z + n_z * Q_y * f_y * x_I * (y_I + y_U) - d_z * z + (Q_y*c_y*y_I*y_U+Q_y*c_y*y_I**2)*n_z


    return [dx_I, dx_U, dy_I, dy_U, dz]

# 상태 분류 함수
def classify_state(solution):
    x_I, x_U, y_I, y_U, z = solution.y[:, -1]  # 마지막 시간 단계 값
    if x_U >= 1 and y_U >= 1 and z >= 1:  # 피식자, 포식자, 기생충 모두 생존
        return 1  # Magenta
    elif x_U >= 1 and y_U >= 1:  # 포식자와 피식자만 생존
        return 2  # Yellow
    elif x_U >= 1:  # 피식자만 생존
        return 3  # Cyan
    else:
        return 0  # 멸종

# 파라미터 범위 설정
Q_y_range = np.linspace(0, 1, 50)
r_p_range = np.linspace(0, 1, 50)
Q_y, r_p = np.meshgrid(Q_y_range, r_p_range)

# 결과 저장 배열 생성
results = np.zeros(Q_y.shape)

# 파라미터 조합에 따라 시뮬레이션 수행
for i in range(Q_y.shape[0]):
    for j in range(Q_y.shape[1]):
        # 초기 조건 및 시간 설정
        initial_conditions = [0, 800, 0, 100, 1000]
        t_span = (0, 150)
        time_points = np.linspace(t_span[0], t_span[1], 1000)

        # 적분 수행
        solution = solve_ivp(system_dynamic, t_span, initial_conditions, args=(Q_y[i, j], r_p[i, j], 1), method='BDF', t_eval=time_points)

        # 상태 분류 및 결과 저장
        if solution.success:
            results[i, j] = classify_state(solution)
        else:
            results[i, j] = 0  # 적분 실패 시 멸종으로 간주

# 색상 지정
cmap = mcolors.ListedColormap(['black', 'magenta', 'yellow', 'cyan'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# pcolormesh를 사용해 결과 시각화
plt.figure(figsize=(8, 6))
plt.pcolormesh(Q_y, r_p, results, cmap=cmap, norm=norm, shading='auto')

plt.xlabel('$Q_x$')
plt.ylabel('$r_p$')
plt.title('Patterns of Species Coexistence, c_y=0.2, Q_x=1')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='magenta', edgecolor='black', label='Predator, prey and parasite'),
    Patch(facecolor='yellow', edgecolor='black', label='Predator and prey'),
    Patch(facecolor='cyan', edgecolor='black', label='Prey'),
]
plt.legend(handles=legend_elements, loc='upper right', title='State', fontsize=12)
plt.show()

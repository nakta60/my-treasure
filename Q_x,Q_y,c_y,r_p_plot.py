import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 시스템 방정식 정의
def system_dynamic(t, vars, r_e, Q_x, Q_y, c_y):
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
    r_p = 1
    k_y = 0.2
    n_z = 6

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
        return 1
    elif x_U >= 1 and y_U >= 1:  # 포식자와 피식자만 생존
        return 2
    elif x_U >= 1:  # 피식자만 생존
        return 3
    elif x_U < 1 and y_U < 1 and z < 1:  # 모든 종이 멸종
        return 0

# 파라미터 범위 설정
Q_x_range = np.linspace(0, 1, 10)
Q_y_range = np.linspace(0, 1, 10)
c_y_range = np.linspace(0, 1, 10)
r_e_range = np.linspace(0, 1, 20)  # `h`에 해당

Q_x, Q_y, c_y = np.meshgrid(Q_x_range, Q_y_range, c_y_range, indexing='ij')

# 결과 저장 배열 생성
H = []  # r_p 값
countzero = []  # 멸종 상태 개수

# 파라미터 조합에 따라 시뮬레이션 수행
for r_e in r_e_range:
    results = np.zeros(Q_x.shape)
    for i in range(Q_x.shape[0]):
        for j in range(Q_x.shape[1]):
            for k in range(Q_x.shape[2]):
                # 초기 조건 및 시간 설정
                initial_conditions = [0, 800, 0, 100, 1000]
                t_span = (0, 150)
                time_points = np.linspace(t_span[0], t_span[1], 1000)

                # 적분 수행
                solution = solve_ivp(
                    system_dynamic,
                    t_span,
                    initial_conditions,
                    args=(r_e, Q_x[i, j, k], Q_y[i, j, k], c_y[i, j, k]),
                    method="BDF",
                    t_eval=time_points,
                )

                # 상태 분류 및 결과 저장
                if solution.success:
                    results[i, j, k] = classify_state(solution)

    # 멸종 상태(0)의 개수 계산
    results_flat = results.ravel()
    count_of_extinction = np.sum(results_flat == 0)  # 멸종 상태(값 0) 카운트
    H.append(r_e)
    countzero.append(count_of_extinction)

# 결과 시각화
plt.plot(H, countzero, marker="o", color="blue", label="Extinction Count")
plt.xlabel("$r_p$ (Reproductive Cost of Predator due to Infection)")
plt.ylabel("Number of Extinct States")
plt.title("Extinction Analysis for $r_p$")
plt.grid()
plt.legend()
plt.show()

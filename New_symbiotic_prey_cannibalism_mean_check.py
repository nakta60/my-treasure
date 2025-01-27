import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors

# 파라미터 범위 설정
f_U_range = np.linspace(0, 1, 200)
f_I_range = np.linspace(0, 1, 200)
f_U, f_I = np.meshgrid(f_U_range, f_I_range)

# 결과 저장 배열 생성
results = np.zeros(f_U.shape)

for i in range(f_U.shape[0]):
    for j in range(f_U.shape[1]):
        cannibalism = []
        def system(t, vars, f_U, f_I):
            x_I, x_U, y_I, y_U, z = vars

            # 논문에서 제시된 매개변수
            K = 2000       # Carrying capacity
            d_x = 0.1      # Death rate of prey
            d_y = 1      # Death rate of predator
            d_z = 0.09      # Death rate of parasite
            Q_x = 1      # Fraction of infected prey contributing to parasite growth
            Q_y = 1      # Fraction of infected predator reproduction
            # f_U = 0.01      # 비감염 피식자를 포식할 확률
            # f_I = 0.001     # 감염된 피식자를 포식할 확률
            S = 0.0005        # Parasite growth rate
            g_U = 2      # 비감염 피식자 번식률
            g_I = 4      # 감염 피식자 번식률
            k_y = 0.2      # Predator growth rate
            r_p = 1      # Reproductive success of infected predator
            r_e = 1      # Reproductive success of uninfected predator
            n_z = 6      # Parasite reproduction rate

            # 동족포식 확률 계산
            T_H = 500  # 피식자 밀도 임계값
            c0 = 0.15  # 기본 동족포식 확률
            alpha = 0.4  # 동족포식 확률 변화폭
            k = 0.1  # 동족포식 변화 속도
            total_prey = x_I + x_U

            # 새로운 동족포식 확률 함수
            c_y = c0 + alpha * np.tanh(k * (T_H - total_prey + 0.5 * (y_U + y_I)))
            c_y = np.clip(c_y, 0, 1)  # 동족포식 확률 제한
            cannibalism.append(c_y)

        # 값 제한 (오버플로우 방지)
            x_I = np.clip(x_I, 0, 1e6)
            x_U = np.clip(x_U, 0, 1e6)
            y_I = np.clip(y_I, 0, 1e6)
            y_U = np.clip(y_U, 0, 1e6)
            z = np.clip(z, 0, 1e6)


            # 미분 방정식 정의
            dx_I = -((x_I + x_U)* x_I) / K - d_x * x_I + Q_x * S * x_U * z - f_I * x_I * (y_I + y_U)
            dx_U = g_U * x_U + g_I * x_I - ((x_I + x_U) * x_U) / K - d_x * x_U - Q_x * S * x_U * z - f_U * x_U * (y_I + y_U)
            dy_I = -d_y * y_I + Q_y * f_I * x_I * y_U - c_y * y_I * y_U + c_y * Q_y * y_I * y_U
            dy_U = f_U * k_y * x_U * (y_U + r_p * y_I) + (r_p * Q_y + r_e * (1 - Q_y)) * f_I * k_y * x_I * y_U + k_y * f_I * x_I * y_I * (r_p**2 * Q_y + r_p * r_e * (1 - Q_y)) - d_y * y_U - c_y * Q_y * y_I * y_U + c_y * k_y * Q_y * r_p * y_I * y_U + r_e * (1 - Q_y) * c_y * k_y * y_I * y_U
            dz = -Q_x * S * x_U * z + n_z * Q_y * f_I * x_I * (y_I + y_U) - d_z * z + n_z * Q_y * c_y * y_I * y_U

            return [dx_I, dx_U, dy_I, dy_U, dz]
        
        # 초기 조건 및 시간 설정
        initial_conditions = [0, 800, 0, 100, 1000]
        t_span = (0, 200)
        time_points = np.linspace(t_span[0], t_span[1], 200000)

        # 적분 수행
        solution = solve_ivp(system, t_span, initial_conditions, args=(f_U[i, j], f_I[i, j]), method='RK45', t_eval=time_points)
        
        # 조건 추가시 필요한 조건들
        x_I, x_U, y_I, y_U, z = solution.y[:,-1]
        total_x = max(0, x_I + x_U)
        total_y = max(0, y_I + y_U)
        z = max(0,z)
        
        # 값 저장장
        results[i, j] = sum(cannibalism)/len(cannibalism)
    
plt.imshow(results, cmap = 'summer', interpolation='nearest', origin='lower', extent=[0,1,0,1])
plt.colorbar(label = 'Cannibalism Probability')
plt.title('Coexsitence and mean of cannibalism probability_Qx=1,Qy=1')
plt.xlabel('f_U')
plt.ylabel('f_I')
plt.show()

        

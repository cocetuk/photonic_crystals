import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
# ==================================================
c = 3e8

eps_A = 2.5
eps_B = 1.0

dA = 0.4
dB = 0.6
a = dA + dB

N_periods = 10          # число периодов
N_realizations = 30    # число реализаций для усреднения

# уровни погрешностей
disorder_levels = [0.0, 0.03, 0.05, 0.10]   # 0%, 3%, 5%, 10%

# ==================================================
# ДИАПАЗОН ЧАСТОТ
# ==================================================
omega_max = 3 * np.pi * c / a
omega = np.linspace(0.01, omega_max, 2500)

x_freq = omega * a / (2*np.pi*c)

# ==================================================
# МАТРИЦА ОДНОГО СЛОЯ
# ==================================================
def layer_matrix(eps, d, omega):
    k = omega / c * np.sqrt(eps)
    return np.array([
        [np.cos(k*d), 1j/np.sqrt(eps)*np.sin(k*d)],
        [1j*np.sqrt(eps)*np.sin(k*d), np.cos(k*d)]
    ], dtype=complex)

# ==================================================
# РАСЧЁТ R(ω) ДЛЯ ОДНОЙ РЕАЛИЗАЦИИ
# ==================================================
def reflection_spectrum(delta):
    """
    delta — относительная погрешность (например 0.05 = 5%)
    """
    R = np.zeros_like(omega)

    for i, w in enumerate(omega):
        M_tot = np.eye(2, dtype=complex)

        for _ in range(N_periods):
            # случайные толщины
            dA_n = dA * (1 + np.random.uniform(-delta, delta))
            dB_n = dB * (1 + np.random.uniform(-delta, delta))

            M_A = layer_matrix(eps_A, dA_n, w)
            M_B = layer_matrix(eps_B, dB_n, w)

            M_tot = M_B @ M_A @ M_tot

        # коэффициент отражения
        num = (M_tot[0,0] + M_tot[0,1]
               - M_tot[1,0] - M_tot[1,1])
        den = (M_tot[0,0] + M_tot[0,1]
               + M_tot[1,0] + M_tot[1,1])

        r = num / den
        R[i] = np.abs(r)**2

    return R

# ==================================================
# ОСНОВНОЙ РАСЧЁТ (УСРЕДНЕНИЕ)
# ==================================================
R_results = {}

for delta in disorder_levels:
    print(f'Расчёт для погрешности {int(delta*100)}%')

    R_avg = np.zeros_like(omega)

    for _ in range(N_realizations):
        R_avg += reflection_spectrum(delta)

    R_avg /= N_realizations
    R_results[delta] = R_avg

# ==================================================
# ГРАФИК: СРАВНЕНИЕ ПОГРЕШНОСТЕЙ
# ==================================================
plt.figure(figsize=(9,5))

for delta, R in R_results.items():
    label = f'{int(delta*100)}%'
    plt.plot(x_freq, R, label=label)

plt.xlabel(r'$\omega a / (2\pi c)$')
plt.ylabel(r'$R(\omega)$')
plt.title('Влияние погрешностей толщин на запрещённую зону')
plt.ylim(-0.02, 1.02)
plt.legend(title='Погрешность')
plt.grid(True)
plt.tight_layout()
plt.show()

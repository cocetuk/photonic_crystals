import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Физические параметры
# -----------------------------
c = 3e8                 # скорость света
eps_A = 12.0            # диэлектрическая проницаемость слоя A
eps_B = 1.0             # диэлектрическая проницаемость слоя B

dA = 0.4                # толщина слоя A
dB = 0.6                # толщина слоя B
a = dA + dB             # период решетки

# Диапазон частот
omega_min = 0.0
omega_max = 2.5 * np.pi * c / a
N_omega = 2000
omega_list = np.linspace(omega_min, omega_max, N_omega)

# -----------------------------
# Вспомогательные функции
# -----------------------------

def layer_matrix(eps, d, omega):
    """
    Матрица распространения в однородном слое
    """
    q = omega / c * np.sqrt(eps)

    return np.array([
        [np.exp(1j * q * d), 0],
        [0, np.exp(-1j * q * d)]
    ], dtype=complex)


def interface_matrix(eps1, eps2):
    """
    Матрица интерфейса между слоями eps1 -> eps2
    (E и dE/dx непрерывны)
    """
    r = np.sqrt(eps2 / eps1)

    return 0.5 * np.array([
        [1 + r, 1 - r],
        [1 - r, 1 + r]
    ], dtype=complex)


# -----------------------------
# Основной расчет
# -----------------------------
k_values = []

for omega in omega_list:
    # Матрицы
    P_A = layer_matrix(eps_A, dA, omega)
    P_B = layer_matrix(eps_B, dB, omega)

    I_AB = interface_matrix(eps_A, eps_B)
    I_BA = interface_matrix(eps_B, eps_A)

    # Полная матрица периода
    T = I_BA @ P_B @ I_AB @ P_A

    # Теорема Блоха
    trace_T = np.trace(T)
    cos_ka = np.real(trace_T) / 2

    # Разрешённые зоны
    if abs(cos_ka) <= 1:
        k = np.arccos(cos_ka) / a
        k_values.append(k)
    else:
        k_values.append(np.nan)

k_values = np.array(k_values)

# -----------------------------
# График зонной структуры
# -----------------------------
plt.figure(figsize=(7, 5))
plt.plot(k_values * a / np.pi, omega_list * a / (2 * np.pi * c), 'k.', markersize=1)

plt.xlabel(r'$k a / \pi$')
plt.ylabel(r'$\omega a / (2\pi c)$')
plt.title('Зонная структура (TMM)')
plt.grid(True)
plt.show()

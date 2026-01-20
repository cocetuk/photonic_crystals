import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
# ==================================================
c = 3e8

eps_A = 14
eps_B = 1

dA = 0.6
dB = 0.4
a = dA + dB

# ==================================================
# ЧИСЛЕННЫЕ ПАРАМЕТРЫ
# ==================================================
omega_max = 2.5 * np.pi * c / a
N_omega = 3000
omega_list = np.linspace(0.01, omega_max, N_omega)

num_zones = 4    # сколько зон Бриллюэна показывать

# ==================================================
# МАТРИЦЫ TMM (бегущие волны)
# ==================================================
def layer_matrix(eps, d, omega):
    q = omega / c * np.sqrt(eps)
    return np.array([
        [np.exp(1j * q * d), 0],
        [0, np.exp(-1j * q * d)]
    ], dtype=complex)

def interface_matrix(eps1, eps2):
    r = np.sqrt(eps2 / eps1)
    return 0.5 * np.array([
        [1 + r, 1 - r],
        [1 - r, 1 + r]
    ], dtype=complex)

# ==================================================
# ОСНОВНОЙ РАСЧЁТ
# ==================================================
k_plot = []
omega_plot = []

for omega in omega_list:

    # Матрицы слоёв
    P_A = layer_matrix(eps_A, dA, omega)
    P_B = layer_matrix(eps_B, dB, omega)

    # Матрицы интерфейсов
    I_AB = interface_matrix(eps_A, eps_B)
    I_BA = interface_matrix(eps_B, eps_A)

    # Матрица одного периода
    T = I_BA @ P_B @ I_AB @ P_A

    # Теорема Блоха
    cos_ka = np.real(np.trace(T)) / 2

    # Разрешённые зоны
    if abs(cos_ka) <= 1:
        ka = np.arccos(cos_ka)

        # Все эквивалентные ветви
        for n in range(num_zones):
            k_plot.append(( ka + 2*np.pi*n ) / a)
            k_plot.append((-ka + 2*np.pi*n ) / a)

            omega_plot.append(omega)
            omega_plot.append(omega)

# ==================================================
# ГРАФИК
# ==================================================
plt.figure(figsize=(9, 6))
plt.plot(
    np.array(k_plot) * a / np.pi,
    np.array(omega_plot) * a / (2*np.pi*c),
    'k.', markersize=0.8
)

plt.xlabel(r'$k a / \pi$', fontsize=12)
plt.ylabel(r'$\omega a / (2\pi c)$', fontsize=12)
plt.title('Зонная структура 1D фотонного кристалла (TMM)', fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.show()



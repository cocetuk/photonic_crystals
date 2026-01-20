import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Физические параметры
# -----------------------------
c = 3e8

eps_A = 12.0
eps_B = 1.0

dA = 0.4
dB = 0.6
a = dA + dB

f = dA / a
G = 2 * np.pi / a

# -----------------------------
# Параметры PWEM
# -----------------------------
M = 7                       # число гармоник (2M+1 волн)
Nk = 200                    # число точек по k
k_list = np.linspace(-np.pi / a, np.pi / a, Nk)

# -----------------------------
# Коэффициенты Фурье κ_m
# -----------------------------
def kappa_m(m):
    """
    Коэффициенты Фурье для 1/eps(x)
    """
    if m == 0:
        return f / eps_A + (1 - f) / eps_B
    else:
        return (1 / (1j * 2 * np.pi * m)) * (1 / eps_A - 1 / eps_B) \
               * (1 - np.exp(-1j * 2 * np.pi * m * f))


# массив κ_{m-m'}
kappa = {}
for m in range(-2*M, 2*M + 1):
    kappa[m] = kappa_m(m)

# -----------------------------
# Основной цикл PWEM
# -----------------------------
bands = []

for k in k_list:
    size = 2 * M + 1
    A = np.zeros((size, size), dtype=complex)

    for i, m in enumerate(range(-M, M + 1)):
        for j, mp in enumerate(range(-M, M + 1)):
            A[i, j] = (k + m * G)**2 * kappa[m - mp]

    # задача на собственные значения
    eigvals = np.linalg.eigvals(A)
    eigvals = np.real(eigvals)

    # оставляем физические собственные значения
    eigvals = eigvals[eigvals > 0]
    omega = c * np.sqrt(eigvals)

    omega.sort()
    bands.append(omega[:6])   # первые 6 зон

bands = np.array(bands)

# -----------------------------
# График зонной структуры
# -----------------------------
plt.figure(figsize=(7, 5))

for n in range(bands.shape[1]):
    plt.plot(k_list * a / np.pi,
             bands[:, n] * a / (2 * np.pi * c),
             'k')

plt.xlabel(r'$k a / \pi$')
plt.ylabel(r'$\omega a / (2\pi c)$')
plt.title('Зонная структура (PWEM)')
plt.grid(True)
plt.show()

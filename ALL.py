import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1. ОБЩИЕ ПАРАМЕТРЫ
# ============================================================================
c = 3e8  # скорость света

# Диэлектрические проницаемости
eps_A = 1.0
eps_B = 5

# Толщины слоев и период
dA = 0.4e-6  # [м]
dB = 0.6e-6  # [м]
a = dA + dB  # период решетки

# Доля слоя A
f = dA / a

# Вектор обратной решетки
G = 2 * np.pi / a


# ============================================================================
# 2. МЕТОД ПЛОСКО-ВОЛНОВОГО РАЗЛОЖЕНИЯ (PWEM)
# ============================================================================
def compute_pwem_bands(eps1, eps2):
    """
    Вычисление зонной структуры методом PWEM
    для заданных диэлектрических проницаемостей
    """
    M = 10  # число гармоник
    Nk = 3000  # число точек по k
    k_list = np.linspace(-np.pi / a, np.pi / a, Nk)

    # Коэффициенты Фурье для 1/eps(x)
    def kappa_m(m, eps1, eps2):
        if m == 0:
            return f / eps1 + (1 - f) / eps2
        else:
            return (1 / (1j * 2 * np.pi * m)) * (1 / eps1 - 1 / eps2) \
                * (1 - np.exp(-1j * 2 * np.pi * m * f))

    # Подготовка массива коэффициентов
    kappa = {}
    for m in range(-2 * M, 2 * M + 1):
        kappa[m] = kappa_m(m, eps1, eps2)

    # Основной цикл PWEM
    pwem_bands = []

    for k in k_list:
        size = 2 * M + 1
        A = np.zeros((size, size), dtype=complex)

        for i, m in enumerate(range(-M, M + 1)):
            for j, mp in enumerate(range(-M, M + 1)):
                A[i, j] = (k + m * G) ** 2 * kappa[m - mp]

        # Собственные значения
        eigvals = np.linalg.eigvals(A)
        eigvals = np.real(eigvals)
        eigvals = eigvals[eigvals > 0]
        omega = c * np.sqrt(eigvals)
        omega.sort()
        pwem_bands.append(omega[:6])  # первые 6 зон

    return np.array(pwem_bands), k_list


# ============================================================================
# 3. МЕТОД ПЕРЕДАЧИ МАТРИЦ (TMM) - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ============================================================================
def compute_tmm_bands():
    """
    Вычисление зонной структуры методом TMM
    с соединением точек только в разрешенных зонах
    """
    # Диапазон частот
    omega_max_pwem = 7.5e15
    omega = np.linspace(1e12, omega_max_pwem, 60000)

    # Волновые числа в средах
    k1 = omega * np.sqrt(eps_A) / c
    k2 = omega * np.sqrt(eps_B) / c

    # Матричный элемент для TMM дисперсии
    X = (
            np.cos(k1 * dA) * np.cos(k2 * dB)
            - 0.5 * (
                    np.sqrt(eps_B / eps_A) + np.sqrt(eps_A / eps_B)
            ) * np.sin(k1 * dA) * np.sin(k2 * dB)
    )

    # Разрешенные зоны
    allowed = np.abs(X) <= 1
    kd = np.arccos(np.clip(X, -1, 1))

    # Создаем массивы с nan для разрывов
    k_plus = kd / a
    k_minus = -kd / a

    # В запрещенных зонах ставим nan, чтобы создать разрывы
    k_plus[~allowed] = np.nan
    k_minus[~allowed] = np.nan

    # Нормированные величины
    k_norm_plus = k_plus * a / np.pi
    k_norm_minus = k_minus * a / np.pi
    omega_norm = omega * a / (2 * np.pi * c)

    return k_norm_plus, k_norm_minus, omega_norm, allowed


def compute_tmm_segments(k_norm_plus, k_norm_minus, omega_norm, allowed):
    """
    Разделение данных TMM на непрерывные сегменты для каждой разрешенной зоны
    """
    # Для положительных k
    segments_plus = []
    current_segment_plus = []
    current_segment_omega_plus = []

    # Для отрицательных k
    segments_minus = []
    current_segment_minus = []
    current_segment_omega_minus = []

    for i in range(len(omega_norm)):
        # Для положительных k
        if allowed[i]:
            current_segment_plus.append(k_norm_plus[i])
            current_segment_omega_plus.append(omega_norm[i])
            current_segment_minus.append(k_norm_minus[i])
            current_segment_omega_minus.append(omega_norm[i])
        else:
            # Заканчиваем текущий сегмент, если он не пустой
            if current_segment_plus:
                segments_plus.append((np.array(current_segment_plus),
                                      np.array(current_segment_omega_plus)))
                segments_minus.append((np.array(current_segment_minus),
                                       np.array(current_segment_omega_minus)))
                current_segment_plus = []
                current_segment_omega_plus = []
                current_segment_minus = []
                current_segment_omega_minus = []

    # Добавляем последний сегмент, если он есть
    if current_segment_plus:
        segments_plus.append((np.array(current_segment_plus),
                              np.array(current_segment_omega_plus)))
        segments_minus.append((np.array(current_segment_minus),
                               np.array(current_segment_omega_minus)))

    return segments_plus, segments_minus

# ============================================================================
# 4. ВЫЧИСЛЕНИЕ ЗАПРЕЩЕННЫХ ЗОН ИЗ TMM
# ============================================================================
def compute_tmm_gaps(k_norm_plus, omega_norm, allowed):
    """
    Вычисление запрещенных зон по данным TMM
    """
    gaps = []
    in_gap = False
    gap_start = 0

    for i in range(len(omega_norm)):
        if not allowed[i] and not in_gap:
            gap_start = omega_norm[i]
            in_gap = True
        if allowed[i] and in_gap:
            gap_end = omega_norm[i]
            gaps.append((gap_start, gap_end))
            in_gap = False

    # Если закончились на запрещенной зоне
    if in_gap:
        gap_end = omega_norm[-1]
        gaps.append((gap_start, gap_end))

    return gaps


# ============================================================================
# 5. ВЫЧИСЛЕНИЯ
# ============================================================================
print("Вычисление PWEM для фотонного кристалла...")
pwem_bands_crystal, k_list = compute_pwem_bands(eps_A, eps_B)

print("Вычисление PWEM для однородной среды...")
eps_eff_hom = f * eps_A + (1 - f) * eps_B
pwem_bands_hom, _ = compute_pwem_bands(eps_eff_hom, eps_eff_hom)

print("Вычисление TMM...")
k_norm_plus, k_norm_minus, omega_norm, allowed = compute_tmm_bands()

print("Вычисление запрещенных зон TMM...")
gaps = compute_tmm_gaps(k_norm_plus, omega_norm, allowed)

# ============================================================================
# 6. ГРАФИК 1: PWEM - кристалл vs однородная среда
# ============================================================================
plt.figure(figsize=(10, 7))

# PWEM для фотонного кристалла
for n in range(pwem_bands_crystal.shape[1]):
    plt.plot(k_list * a / np.pi,
             pwem_bands_crystal[:, n] * a / (2 * np.pi * c),
             'b-', linewidth=2, alpha=0.8)

# PWEM для однородной среды
for n in range(pwem_bands_hom.shape[1]):
    plt.plot(k_list * a / np.pi,
             pwem_bands_hom[:, n] * a / (2 * np.pi * c),
             'r--', linewidth=2, alpha=0.7)

plt.xlabel(r'$k a / \pi$', fontsize=14)
plt.ylabel(r'$\omega a / (2\pi c)$', fontsize=14)
plt.title('Сравнение PWEM: фотонный кристалл и однородная среда', fontsize=16)
plt.grid(True, alpha=0.3)

# Легенда
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='b', linewidth=2, label='Фотонный кристалл (PWEM)'),
    Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Однородная среда (PWEM)')
]
plt.legend(handles=legend_elements, fontsize=12, loc='upper left')

plt.xlim(-1, 1)
plt.ylim(0, min(pwem_bands_crystal[-1, 5] * a / (2 * np.pi * c),
                pwem_bands_hom[-1, 5] * a / (2 * np.pi * c)) + 0.45)
plt.tight_layout()
plt.show()

# ============================================================================
# 7. ГРАФИК 2: PWEM vs TMM (с непрерывными линиями только в разрешенных зонах)
# ============================================================================
plt.figure(figsize=(10, 7))

# Запрещенные зоны от TMM (фоном)
for start, end in gaps:
    plt.axhspan(start, end, color='gray', alpha=0.15, label='Запрещенные зоны (TMM)' if start == gaps[0][0] else '')

# PWEM для фотонного кристалла (линии)
for n in range(pwem_bands_crystal.shape[1]):
    plt.plot(k_list * a / np.pi,
             pwem_bands_crystal[:, n] * a / (2 * np.pi * c),
             'b-', linewidth=2.5, alpha=0.8)

# TMM - непрерывные линии только в разрешенных зонах
segments_plus, segments_minus = compute_tmm_segments(k_norm_plus, k_norm_minus, omega_norm, allowed)

# Рисуем каждый сегмент отдельной линией
for k_segment, omega_segment in segments_plus:
    plt.plot(k_segment, omega_segment, 'r-', linewidth=2, alpha=0.7, zorder=2)

for k_segment, omega_segment in segments_minus:
    plt.plot(k_segment, omega_segment, 'r-', linewidth=2, alpha=0.7, zorder=2)

# Добавим точки для лучшей видимости
allowed_indices = allowed
plt.scatter(k_norm_plus[allowed_indices], omega_norm[allowed_indices],
            color='red', s=3, alpha=0.5, zorder=3)
plt.scatter(k_norm_minus[allowed_indices], omega_norm[allowed_indices],
            color='red', s=3, alpha=0.5, zorder=3)

plt.xlabel(r'$k a / \pi$', fontsize=14)
plt.ylabel(r'$\omega a / (2\pi c)$', fontsize=14)
plt.title('Сравнение PWEM и TMM методов (непрерывные линии TMM только в разрешенных зонах)', fontsize=16)
plt.grid(True, alpha=0.3)

# Легенда
legend_elements = [
    Line2D([0], [0], color='b', linewidth=2.5, label='PWEM'),
    Line2D([0], [0], color='r', linewidth=2, label='TMM'),
    plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.15, label='Запрещенные зоны (TMM)')
]
plt.legend(handles=legend_elements, fontsize=9, loc='upper left')

plt.xlim(-1, 1)
plt.ylim(0, min(pwem_bands_crystal[-1, 5] * a / (2 * np.pi * c) + 0.25, omega_norm[-1]))
plt.tight_layout()
plt.show()
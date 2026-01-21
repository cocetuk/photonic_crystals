import numpy as np
import matplotlib.pyplot as plt

# ============================
# Параметры (можешь менять)
# ============================
c = 3e8                 # скорость света (м/с)
eps_A = 2.5        # eps слоя A
eps_B = 1             # eps слоя B
dA = 0.4             # толщина слоя A (м)
dB = 0.6               # толщина слоя B (м)
a = dA + dB             # период

N_periods = 10         # число периодов конечного кристалла

# частотная сетка
omega_max = 5 * np.pi * c / a
N_omega = 3000
omega = np.linspace(0.01, omega_max, N_omega)

# малый допуск для численной проверки |cos(ka)| <= 1
tol = 1e-12

# ============================
# Функции для матриц TMM
# ============================
def layer_matrix(eps, d, omega):
    """Матрица слоя в (E,H)-представлении (TE, нормальное падение)."""
    k = omega / c * np.sqrt(eps)
    return np.array([
        [np.cos(k*d), 1j/np.sqrt(eps)*np.sin(k*d)],
        [1j*np.sqrt(eps)*np.sin(k*d), np.cos(k*d)]
    ], dtype=complex)

# ============================
# Основной расчёт
# ============================
R = np.zeros_like(omega)           # коэффициент отражения
k_eff = np.zeros_like(omega, dtype=complex)  # эффективный k (комплексный)
allowed = np.zeros_like(omega, dtype=bool)   # маска: разрешённая зона (вещественный k)

for idx, w in enumerate(omega):
    # матрицы слоёв
    M_A = layer_matrix(eps_A, dA, w)
    M_B = layer_matrix(eps_B, dB, w)
    # матрица одного периода
    M_period = M_B @ M_A

    # cos(ka) — может быть комплексным из-за численных неточностей; сохраняем комплексность
    cos_ka = np.trace(M_period) / 2.0

    # эффективный ka (разрешаем комплексное значение)
    # arccos на комплексных аргументах даёт комплексный результат
    ka = np.arccos(cos_ka + 0j)         # +0j гарантирует комплексный путь
    k_eff[idx] = ka / a

    # пометим разрешённую зону: если |Re(cos_ka)| <= 1 и Im(cos_ka) ~ 0
    # более строгий критерий: |cos_ka.real| <= 1 и |cos_ka.imag| < small_tol
    # но достаточно проверить abs(real)<=1 с допуском
    if np.abs(np.real(cos_ka)) <= 1.0 + 1e-10 and np.abs(np.imag(cos_ka)) < 1e-6:
        allowed[idx] = True
    else:
        allowed[idx] = False

    # матрица всего конечного кристалла
    M_period = M_B @ M_A
    M_tot = np.linalg.matrix_power(M_period, N_periods)

    # коэффициент отражения (амплитудный r, затем R=|r|^2)
    # формула для нормального падения и одинаковых внешних сред (eps=1)
    num = (M_tot[0,0] + M_tot[0,1] - M_tot[1,0] - M_tot[1,1])
    den = (M_tot[0,0] + M_tot[0,1] + M_tot[1,0] + M_tot[1,1])
    r = num / den
    R[idx] = np.abs(r)**2

# ============================
# Подготовка данных для рисования Re(k) с "разрывами"
# ============================
# мы хотим, чтобы линия Re(ka/pi) не шла через запрещённые зоны:
re_ka_pi = np.real(k_eff) * a / np.pi
# маскируем (ставим NaN) в запрещённых местах — matplotlib не соединит через NaN
re_ka_pi_masked = re_ka_pi.copy()
re_ka_pi_masked[~allowed] = np.nan

# Im(ka/pi) (в запрещённых зонах будет > 0)
im_ka_pi = np.imag(k_eff) * a / np.pi

# ============================
# Находим интервалы запрещённых зон (для затемнения на общем графике)
# ============================
forbidden_mask = ~allowed  # True где запрещено (Im(k) > 0 обычно)
# найдем непрерывные сегменты True
forbidden_intervals = []
in_interval = False
start_idx = None
for i, val in enumerate(forbidden_mask):
    if val and not in_interval:
        in_interval = True
        start_idx = i
    elif not val and in_interval:
        end_idx = i - 1
        forbidden_intervals.append((start_idx, end_idx))
        in_interval = False
# если дошли до конца и остались в интервале
if in_interval:
    forbidden_intervals.append((start_idx, len(forbidden_mask)-1))

# ============================
# Построение графиков
# ============================
x_freq = omega * a / (2.0 * np.pi * c)   # нормированная частота ω a / (2π c)

# 1) R(ω)
plt.figure(figsize=(8,4.5))
plt.plot(x_freq, R, color='blue')
plt.ylim(-0.02, 1.02)
plt.xlabel(r'$\omega a / (2\pi c)$')
plt.ylabel(r'$R(\omega)$')
plt.title('Коэффициент отражения конечного ФК')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Re(ka/pi) (с разрывами)
plt.figure(figsize=(8,4.5))
plt.plot(re_ka_pi_masked, x_freq, color='blue')   # обратите внимание: оси меняем местами для удобства
plt.xlabel(r'$\mathrm{Re}(k a / \pi)$')
plt.ylabel(r'$\omega a / (2\pi c)$')
plt.title('Разрешённые зоны: $\mathrm{Re}(k a / \pi)$')
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) Im(ka/pi)
plt.figure(figsize=(8,4.5))
plt.plot(x_freq, im_ka_pi, color='black')
plt.xlabel(r'$\omega a / (2\pi c)$')
plt.ylabel(r'$\mathrm{Im}(k a / \pi)$')
plt.title('Запрещённая зона: $\mathrm{Im}(k a / \pi)$ (затухание)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) Общий график: наложение R(ω) и Re(ka/pi) + затемнение запрещённых зон
fig, ax1 = plt.subplots(figsize=(9,5))

# plot R on left y
ax1.plot(x_freq, R, label='R(ω)', color='tab:blue')
ax1.set_xlabel(r'$\omega a / (2\pi c)$')
ax1.set_ylabel(r'$R(\omega)$', color='tab:blue')
ax1.set_ylim(-0.02, 1.02)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# shade forbidden frequency bands
for (s_idx, e_idx) in forbidden_intervals:
    x0 = x_freq[s_idx]
    x1 = x_freq[e_idx]
    ax1.axvspan(x0, x1, color='gray', alpha=0.25)

# plot Re(ka/pi) on right y (but as function x_freq)
ax2 = ax1.twinx()
ax2.plot(x_freq, re_ka_pi, label=r'Re$(ka/\pi)$', color='tab:red')
ax2.set_ylabel(r'Re$(ka/\pi)$', color='tab:red')

# Выравнивание осей так, чтобы 0 и 1 совпадали по вертикали
# Получаем текущие пределы для обеих осей
y1_min, y1_max = ax1.get_ylim()  # Обычно это -0.02, 1.02
y2_min, y2_max = ax2.get_ylim()  # Текущие пределы для второй оси

# Находим коэффициенты преобразования для второй оси
# Чтобы совместить точки 0 и 1 на обеих осях
# y2_new = a*y2 + b, где:
# a*y2_min + b = y1_min  (для точки 0)
# a*y2_max + b = y1_max  (для точки 1)

# Но у нас на левой оси значения 0 и 1 соответствуют определенным координатам
# Пусть значение 0 на правой оси соответствует значению 0 на левой оси
# и значение 1 на правой оси соответствует значению 1 на левой оси

# Находим значения на правой оси, которые соответствуют 0 и 1 по величине
# (не по координатам на графике)
y2_at_R0 = 0  # Значение Re(ka/pi), соответствующее R=0
y2_at_R1 = 1  # Значение Re(ka/pi), соответствующее R=1

# Теперь нам нужно найти такие пределы для правой оси,
# чтобы y2_at_R0 оказался на той же высоте, что и 0 на левой оси,
# и y2_at_R1 на той же высоте, что и 1 на левой оси

# Это линейное преобразование: y1 = m * y2 + b
# Для двух точек:
# 0 = m * y2_at_R0 + b  (1)
# 1 = m * y2_at_R1 + b  (2)

# Вычитаем (1) из (2): 1 = m * (y2_at_R1 - y2_at_R0)
m = 1.0 / (y2_at_R1 - y2_at_R0)
b = -m * y2_at_R0

# Теперь находим новые пределы для правой оси, которые дадут
# те же визуальные пределы, что и у левой оси (-0.02, 1.02)
new_y2_min = (y1_min - b) / m
new_y2_max = (y1_max - b) / m

# Устанавливаем новые пределы для правой оси
ax2.set_ylim(new_y2_min, new_y2_max)
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Сравнение: R(ω) и Re(ka/π). Заштрихованы запрещённые зоны')
fig.tight_layout()
plt.show()

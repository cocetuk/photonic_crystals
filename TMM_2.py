import numpy as np
import matplotlib.pyplot as plt

eps1 = 1.0
eps2 = 2.5
d1 = 0.1
d2 = 0.2
a = d1 + d2
c = 1.0


omega = np.linspace(0.01, 20, 5000)

kB_list = []
omega_list = []

for w in omega:
    k1 = w * np.sqrt(eps1) / c
    k2 = w * np.sqrt(eps2) / c

    M1 = np.array([
        [np.cos(k1*d1), 1j/np.sqrt(eps1)*np.sin(k1*d1)],
        [1j*np.sqrt(eps1)*np.sin(k1*d1), np.cos(k1*d1)]
    ])

    M2 = np.array([
        [np.cos(k2*d2), 1j/np.sqrt(eps2)*np.sin(k2*d2)],
        [1j*np.sqrt(eps2)*np.sin(k2*d2), np.cos(k2*d2)]
    ])

    M = M2 @ M1
    tr = np.real(np.trace(M) / 2)

    if abs(tr) <= 1:
        kB = np.arccos(tr) / a
        kB_list.append(kB)
        omega_list.append(w)


kB = np.array(kB_list)
omega = np.array(omega_list)

plt.figure(figsize=(8,6))
plt.plot(kB, omega, 'b', lw=1)
plt.plot(-kB, omega, 'b', lw=1)

plt.xlabel(r'$k_B$')
plt.ylabel(r'$\omega$ (norm.)')
plt.title('1D Photonic Crystal Dispersion')
plt.grid(True)
plt.show()

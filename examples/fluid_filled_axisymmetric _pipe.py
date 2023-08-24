import numpy as np
import matplotlib.pyplot as plt
import pywfe

model = pywfe.load(
    "20-21cm axisymmetric water filled steel shell", source='database')


# %%

model.see()

# %%

f_arr = np.linspace(100, 10000, 200)
cp = model.phase_velocity(f_arr, imag_threshold=0.5)

# %%

plt.plot(f_arr, cp, '.')


# %%

k = model.frequency_sweep(f_arr, quantities=['wavenumbers'], mac=True)

# %%

plt.plot(f_arr, k['wavenumbers'])

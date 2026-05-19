import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# 1. YOUR DATA
raw_data = [1189,-215], [1176,-339], [988,-432], [960,-522], [960,-596], [837,-674], [665,-801], [549,-921], [478,-1047], [483,-1162], [505,-1273], [411,-1374], [366,-1487], [363,-1616], [432,-1679], [539,-1703], [761,-1700], [991,-1678], [850,-1626], [656,-1619], [483,-1583], [464,-1528], [507,-1452], [558,-1399], [623,-1495], [1017,-1487], [988,-1441], [927,-1429], [1032,-1363], [1140,-1496], [1278,-1487], [1235,-1424], [1171,-1415], [1168,-1211], [1175,-1010], [1218,-936], [1273,-843], [1264,-713], [1276,-653], [1286,-595], [1337,-565], [1382,-513], [1385,-458], [1349,-435], [1321,-436], [1293,-393], [1242,-356], [1191,-361], [1206,-297], [1212,-256], 

points = np.array(raw_data)
x_vals = points[:, 0]
y_vals = points[:, 1]

# 2. PRE-PROCESSING
# Centering the data so it rotates around the origin
x_vals = x_vals - np.mean(x_vals)
y_vals = y_vals - np.mean(y_vals)

# Interpolate to create a smooth path (from 40 points to 500)
t = np.linspace(0, 1, len(points))
t_smooth = np.linspace(0, 1, 500)
# 'slinear' or 'cubic' works well for your sharp angles
x_smooth = interp1d(t, x_vals, kind='linear')(t_smooth)
y_smooth = interp1d(t, y_vals, kind='linear')(t_smooth)

# Convert to complex numbers
z = x_smooth + 1j * y_smooth

# 3. FOURIER TRANSFORM
n = len(z)
coeffs = np.fft.fft(z) / n
freqs = np.fft.fftfreq(n, d=1/n)

# Sort by amplitude to make the epicycles look "ordered"
idx = np.argsort(np.abs(coeffs))[::-1]
coeffs = coeffs[idx]
freqs = freqs[idx]

# 4. ANIMATION SETUP
fig, ax = plt.subplots(figsize=(5, 6))
ax.set_aspect('equal')
# Set limits based on your data spread
ax.set_xlim(np.min(x_vals)-100, np.max(x_vals)+100)
ax.set_ylim(np.min(y_vals)-100, np.max(y_vals)+100)
ax.axis('off')

drawing, = ax.plot([], [], color='#2c3e50', lw=2)  # The path
arm, = ax.plot([], [], color='#e74c3c', alpha=0.4, marker='o', markersize=2) # The epicycles
trace_x, trace_y = [], []

def update(frame):
    dt = frame / n
    current_pos = 0 + 0j
    cx, cy = [0], [0]
    
    # Using the first 100 circles for high detail
    for i in range(min(100, len(coeffs))):
        current_pos += coeffs[i] * np.exp(1j * 2 * np.pi * freqs[i] * dt)
        cx.append(current_pos.real)
        cy.append(current_pos.imag)
        
    trace_x.append(current_pos.real)
    trace_y.append(current_pos.imag)
    
    drawing.set_data(trace_x, trace_y)
    arm.set_data(cx, cy)
    return drawing, arm

ani = FuncAnimation(fig, update, frames=n, interval=20, blit=True)
plt.title("Fourier Reconstruction of Your Coordinates")
plt.show()

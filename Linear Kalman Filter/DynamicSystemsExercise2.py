import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

m = 1
b = 1.0
k = 10.0

A = np.array([[-b/m, -k/m],[1,0]])
B = np.array([1/m, 0])
f = 1.0
dt = 0.1
tEnd = 10.0

F = expm(A*dt)
G = np.matmul(F,np.matmul((np.identity(2) - expm(-A*dt)),np.matmul(np.linalg.inv(A), B)))

T = 0.0
X = np.array([0,0])
Time = [T]
Velocity = [X[0]]
Position = [X[1]]

while (T < tEnd):
  T = T + dt
  X = np.matmul(F,X) + G*f
  Time.append(T)
  Velocity.append(X[0])
  Position.append(X[1])

fig, axs = plt.subplots(2)
fig.suptitle('System Response\n2nd Order Spring-Damper System')
axs[0].plot(Time, Position)
axs[0].set_ylabel("Position (m)")
axs[0].grid(True)
axs[1].plot(Time, Velocity)
axs[1].set_ylabel("Velocity (m/s)")
axs[1].set_xlabel("Time (s)")
axs[1].grid(True)

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from Functions import generate_data, kalman_filter, rts_smoothing
from starman import kalman, KalmanFilter

### Data generator ###

d = 4
p = 1
m = 2
T = 500
dt = 0.1

F = np.eye((d))
F[0,2] = dt
F[1,3] = dt

B = np.array([0,-0.5*dt**2, 0 ,-0.5*dt**2 ])

u=9.8

Q = np.eye((d)) * 0.01

R = np.eye((2)) * 100

X = np.zeros((T, 4))
X[0, :] = np.array([10 , 1000, 5, 10 ])

v = np.random.multivariate_normal(np.array([0,0]), R, T).T

H = np.zeros((2, 4))
H[0,0] = 1
H[1,1] = 1

X, Y = generate_data(X=X, F=F, B=B, u=u, Q=Q, H=H, v=v, T=T)

plt.scatter(X[0, :], X[1, :], s=1, color='crimson')
plt.scatter(Y[0, :], Y[1, :], s=1, color='blue')

### Kalman filter ###

Y = np.transpose(Y)
X = np.array([10, 1000, 5, 10]).T
Sigma = np.zeros((4, 4))

X_kk, Sigma_kk = kalman_filter(X=X, Y=Y, F=F, B=B, u=u, Q=Q, H=H, R=R, Sigma=Sigma, T=T)

print(F)
print(X_kk)

plt.scatter(X_kk[1:-1, 0], X_kk[1:-1, 1], s=1, color='green')
plt.show()

smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk, Sigma_kk, F, Q)

plt.scatter(X_kk[1:-1, 0], X_kk[1:-1, 1], s=1, color='green')
plt.scatter(smooth_x[1:-1, 0], smooth_x[1:-1, 1], s=1, color='black')
plt.show()



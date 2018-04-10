import numpy as np
import matplotlib.pyplot as plt
from Functions import generate_data, kalman_filter, rts_smoothing, generate_QR_EM
from starman import kalman, KalmanFilter

### Data generator ###

d = 4; p = 1; m = 2; T = 500; dt = 0.1
F = np.eye((d)); F[0,2] = dt; F[1,3] = dt
B = np.array([0,-0.5*dt**2, 0 ,-0.5*dt**2 ])
u=9.8
Q = np.eye((d)) * 0.01
R = np.eye((m)) * 100
X0=np.array([10 , 1000, 5, 10 ]); X = np.zeros((T, 4)); X[0, :] = X0
H = np.zeros((m, d)); H[0,0] = 1; H[1,1] = 1

X, Y = generate_data(X=X, F=F, B=B, u=u, Q=Q, H=H, R=R, T=T)
plt.scatter(X[0, :], X[1, :], s=1, color='crimson')
plt.scatter(Y[0, :], Y[1, :], s=1, color='blue')
X=X.T
Y=Y.T


### Kalman filter ###

Sigma = np.zeros((4, 4))

X_kk, Sigma_kk = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q, H=H, R=R, Sigma=Sigma, T=T)

# plt.scatter(X_kk[1:-1, 0], X_kk[1:-1, 1], s=1, color='green')
# plt.show()

#rts smoothing#
smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk, Sigma_kk, F, Q)

# plt.scatter(X_kk[1:-1, 0], X_kk[1:-1, 1], s=1, color='green')
# plt.scatter(smooth_x[1:-1, 0], smooth_x[1:-1, 1], s=1, color='black')
# plt.show()



## Repeat Kalman for Q and R from EM
Q_init=Q
R_init=R
Q_EM=np.random.rand(d,d)
R_EM=np.random.rand(m,m)
for j in range(200):
    X_kk, Sigma_kk = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q_EM, H=H, R=R_EM, Sigma=Sigma, T=T)
    smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk, Sigma_kk, F, Q_EM)
    Q_EM, R_EM = generate_QR_EM(smooth_x, smooth_sig, L_smooth, X_kk, Y, F, H)
print("Qem",Q_EM)
print("Q", Q_init)
print("Rem",R_EM)
print("R", R_init)



# # data with estimated Q R
# v_EM = np.random.multivariate_normal(np.array([0,0]), R_EM, T).T
# X_EM, Y_EM = generate_data(X=X, F=F, B=B, u=u, Q=Q_EM, H=H, v=v_EM, T=T)
#
# # plt.scatter(X_EM[0, :], X_EM[1, :], s=1, color='crimson')
# # plt.scatter(Y_EM[0, :], Y_EM[1, :], s=1, color='blue')
#
# Y_EM=Y_EM.T
# X = np.array([10, 1000, 5, 10]).T
# Sigma = np.zeros((4, 4))
#
# X_kk_EM, Sigma_kk_EM = kalman_filter(X=X, Y=Y_EM, F=F, B=B, u=u, Q=Q_EM, H=H, R=R_EM, Sigma=Sigma, T=T)
#
# #
# # plt.scatter(X_kk_EM[1:-1, 0], X_kk_EM[1:-1, 1], s=1, color='green')
# # plt.show()


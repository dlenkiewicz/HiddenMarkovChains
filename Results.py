import numpy as np
from pykalman.standard import KalmanFilter
import matplotlib.pyplot as plt
from Functions import generate_data, kalman_filter, rts_smoothing, generate_QR_EM

### Data generator ###

d = 4;
p = 1;
m = 2;
T = 500;
dt = 0.1
F = np.eye((d));
F[0, 2] = dt;
F[1, 3] = dt

B = np.array([0, -0.5 * dt ** 2, 0, -0.5 * dt ** 2])
u = 9.8
Q = np.eye((d)) * 0.01
R = np.eye((m)) * 100

mu = [0, 0, 0, 0]
Sigma = np.zeros((4, 4))
print(Sigma)
X0 = np.array([10, 1000, 5, 10])

X = np.zeros((T, 4));
X[0, :] = X0
print(X[0])

H = np.zeros((m, d));
H[0, 0] = 1;
H[1, 1] = 1

X, Y = generate_data(X=X, F=F, B=B, u=u, Q=Q, H=H, R=R, T=T)

# plt.scatter(X[0, :], X[1, :], s=1, color='crimson')
# plt.scatter(Y[0, :], Y[1, :], s=1, color='blue')
X = X.T
Y = Y.T

superKalman = KalmanFilter(transition_matrices=F, observation_matrices=H, transition_offsets=B * u,
                           initial_state_mean=X0, initial_state_covariance=Sigma,
                           em_vars=['transition_covariance', 'observation_covariance'])
superKalman.em(X=Y, n_iter=1)
print("Nie nasze Q", superKalman.transition_covariance)
print("Nie Nasz R", superKalman.observation_covariance)

### Kalman filter ###


# X_kk, Sigma_kk = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q, H=H, R=R, Sigma=Sigma, T=T)

# plt.scatter(X_kk[1:-1, 0], X_kk[1:-1, 1], s=1, color='green')
# plt.show()

# rts smoothing#
# smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk, Sigma_kk, F, Q)

# plt.scatter(X_kk[1:-1, 0], X_kk[1:-1, 1], s=1, color='green')
# plt.scatter(smooth_x[1:-1, 0], smooth_x[1:-1, 1], s=1, color='black')
# plt.show()


## Repeat Kalman for Q and R from EM
Q_init = Q
R_init = R
Q_EM = np.eye((d))
R_EM = np.eye((m))
for j in range(40):
    Q = Q_EM
    R = R_EM
    X_kk, Sigma_kk = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q, H=H, R=R, Sigma=Sigma, T=T)
    smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk, Sigma_kk, F, Q)
    Q_EM, R_EM = generate_QR_EM(smooth_x, smooth_sig, L_smooth, X_kk, Y, F, H, B*u)
print("Nasz Qem", Q_EM)
print("Nasze Rem", R_EM)

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
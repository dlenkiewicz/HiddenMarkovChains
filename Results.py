import numpy as np
from pykalman.standard import KalmanFilter
import matplotlib.pyplot as plt
from sklearn import metrics
from Functions import generate_data, kalman_filter, rts_smoothing, generate_QR_EM

### Data generator ###

d = 4
p = 1
m = 2
T = 500
dt = 0.1
F = np.eye((d))
F[0, 2] = dt
F[1, 3] = dt

B = np.array([0, -0.5 * dt ** 2, 0, -0.5 * dt ** 2])
u = 9.8
Q = np.eye((d)) * 0.01
R = np.eye((m)) * 100

mu = [0, 0, 0, 0]
Sigma = np.zeros((4, 4))
X0 = np.array([10, 1000, 5, 10])

X = np.zeros((T, 4));
X[0, :] = X0

H = np.zeros((m, d));
H[0, 0] = 1;
H[1, 1] = 1

X, Y = generate_data(X=X, F=F, B=B, u=u, Q=Q, H=H, R=R, T=T)
X = X.T
Y = Y.T

plt.scatter(X[:, 0], X[:, 1], s=1, color='crimson', label="X")
plt.scatter(Y[:, 0], Y[:, 1], s=1, color='blue', label="Y")

### Kalman filter ###

X_kk, Sigma_kk = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q, H=H, R=R, Sigma=Sigma, T=T)

print("MSE - Kalman filter:", metrics.mean_squared_error(X_kk[:, 0], X[:, 0]))

plt.scatter(X_kk[:, 0], X_kk[:, 1], s=1, color='green', label="est. X")

### RTS smoothing ###

smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk, Sigma_kk, F, Q)

print("MSE - RTS smoothing:", metrics.mean_squared_error(smooth_x[:, 0], X[:, 0]))

plt.scatter(smooth_x[:, 0], smooth_x[:, 1], s=1, color='black', label="smoothed est. X")
plt.legend(loc='best', ncol=2).get_frame().set_alpha(0.5)
plt.show()

### EM ###

Q_init = Q
R_init = R
Q_EM = np.eye((d))
R_EM = np.eye((m))
X_kk_em = X_kk
Sigma_kk_em = Sigma_kk

for j in range(40):
    Q = Q_EM
    R = R_EM
    X_kk_em, Sigma_kk_em = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q, H=H, R=R, Sigma=Sigma, T=T)
    smooth_x, smooth_sig, L_smooth = rts_smoothing(X_kk_em, Sigma_kk_em, F, Q)
    Q_EM, R_EM = generate_QR_EM(smooth_x, smooth_sig, L_smooth, X_kk, Y, F, H, B*u)

# print("Q EM", Q_EM)
# print("R EM", R_EM)

X_kk_em, Sigma_kk_em = kalman_filter(X=X0, Y=Y, F=F, B=B, u=u, Q=Q_EM, H=H, R=R_EM, Sigma=Sigma, T=T)

print("MSE - EM - Q:", metrics.mean_squared_error(Q_init, Q_EM))
print("MSE - EM - R:", metrics.mean_squared_error(R_init, R_EM))

plt.scatter(X_kk[:, 0], X_kk[:, 1], s=1, color='green', label="est. X with theor. Q and R")
plt.scatter(X_kk_em[:, 0], X_kk_em[:, 1], s=1, color='orange', label="est. X with EM Q and R")
plt.legend(loc='best', ncol=2).get_frame().set_alpha(0.5)
plt.show()

### implemented EM ###

# superKalman = KalmanFilter(transition_matrices=F, observation_matrices=H, transition_offsets=B * u,
#                            initial_state_mean=X0, initial_state_covariance=Sigma,
#                            em_vars=['transition_covariance', 'observation_covariance'])
# superKalman.em(X=Y, n_iter=40)
# print("implemented Q:", superKalman.transition_covariance)
# print("implemented R:", superKalman.observation_covariance)
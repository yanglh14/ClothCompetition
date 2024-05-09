# script to testing the trajectory of left and right robots
#
import numpy as np
import matplotlib.pyplot as plt

POSI_start_L = [0,0,0.9]
POSI_start_R = [0,0,0.1]
L = POSI_start_L[2] - POSI_start_R[2]
N = 30

X0_L, Y0_L, Z0_L = POSI_start_L
X0_R, Y0_R, Z0_R = POSI_start_R
Y_R_ls,Z_R_ls,Y_L_ls,Z_L_ls = [],[],[],[]

for i in range(N+1):
    X_L = X0_L
    Y_L = Y0_L - i*L/(2*N)
    Z_L = Z0_L
    theta = i*np.pi/(2*N)
    X_R = X0_R
    Y_R = Y_L + L*np.sin(theta)
    Z_R = Z_L - L*np.cos(theta)

    POSI_L = np.array([X_L, Y_L, Z_L])
    POSI_R = np.array([X_R, Y_R, Z_R])
    Y_L_ls.append(Y_L)
    Z_L_ls.append(Z_L)
    Y_R_ls.append(Y_R)
    Z_R_ls.append(Z_R)

    # cal. distance of two grippers
    dist = np.linalg.norm(POSI_L - POSI_R)
    print(f'dist[#{i}]: {dist}')

# plot trajectory
plt.plot(Y_L_ls, Z_L_ls, 'ro')
plt.plot(Y_R_ls, Z_R_ls, 'bo')
plt.xlabel('Y')
plt.ylabel('Z')
plt.xlim(-0.5,0.5)
plt.ylim(0,1)
plt.grid()
# square plot
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
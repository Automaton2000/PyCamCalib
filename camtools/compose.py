import sys
import numpy as np


def qr3(A):
    # Hartley, Zisserman p.579

    c = -A[2, 2] / np.sqrt(A[2, 2] * A[2, 2] + A[2, 1] * A[2, 1])
    s = A[2, 1] / np.sqrt(A[2, 2] * A[2, 2] + A[2, 1] * A[2, 1])
    Q_x = np.array([[1, 0, 0],
                    [0, c, -s],
                    [0, s, c]])
    R = A @ Q_x

    c = R[2, 2] / np.sqrt(R[2, 2] * R[2, 2] + R[2, 0] * R[2, 0])
    s = R[2, 0] / np.sqrt(R[2, 2] * R[2, 2] + R[2, 0] * R[2, 0])
    Q_y = np.array([[c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]])
    R = R @ Q_y

    c = -R[1, 1] / np.sqrt(R[1, 1] * R[1, 1] + R[1, 0] * R[1, 0])
    s = R[1, 0] / np.sqrt(R[1, 1] * R[1, 1] + R[1, 0] * R[1, 0])
    Q_z = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])
    R = R @ Q_z

    Q = Q_z.T @ Q_y.T @ Q_x.T

    for i in range(0,3):
        if R[i, i]<0:
            R[:, i] = -R[:, i]
            Q[i, :] = -Q[i, :]

    return Q, R


def decompose_camera(P):

    # compute the camera center
    T = -np.linalg.det(np.c_[P[:, 0], P[:, 1], P[:, 2]])
    X = np.linalg.det(np.c_[P[:, 1], P[:, 2], P[:, 3]])
    Y = -np.linalg.det(np.c_[P[:, 0], P[:, 2], P[:, 3]])
    Z = np.linalg.det(np.c_[P[:, 0], P[:, 1], P[:, 3]])

    C = np.c_[X, Y, Z, T]   # camera center in homogeneous coordinates
    C_tilde = C[0, 0:3] / C[0, 3]
    print(C_tilde)

    # principal point computation
    M = np.c_[P[:, 0], P[:, 1], P[:, 2]]
    pp = M @ M[2, :]
    pp_tilde = pp[0:2] / pp[2]
    print(pp_tilde)

    #
    R, K = qr3(M)

    return K, R, pp_tilde, C_tilde


def compose_camera(a_x, a_y, s, pp, R, C):

    K = np.array([[a_x, s, pp[0]],
                  [0, a_y, pp[1]],
                  [0, 0, 1]])
    t = -R @ C
    print(t)
    M = np.c_[R, t]
    print(M)
    P = K @ M

    return P

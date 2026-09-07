import numpy as np
from auxiliary_functions import epipolar_line

def skew(v):
    """
    Antisymetric Matrix [v]x
    """
    x, y, z = v.flatten()

    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

def relative_pose(cal1, cal2):
    """
    Calcule a relative pose between two cameras.

    Retorna:
        R_rel
        t_rel
    """

    R1 = cal1["E"][:3, :3]
    t1 = cal1["E"][:3, 3].reshape(3,1)

    R2 = cal2["E"][:3, :3]
    t2 = cal2["E"][:3, 3].reshape(3,1)

    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    return R_rel, t_rel

def essential_matrix(cal1, cal2):

    R_rel, t_rel = relative_pose(cal1, cal2)

    return skew(t_rel) @ R_rel

def fundamental_matrix(cal1, cal2):

    K1 = cal1["K"]
    K2 = cal2["K"]

    E = essential_matrix(cal1, cal2)

    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    return F

def point_line_distance(line, point):
    """
    Distance between a point and a line.

    line = (a,b,c)
    point = (x,y)
    """

    a, b, c = line
    x, y = point

    return abs(a*x + b*y + c) / np.sqrt(a*a + b*b)

def skeleton_epipolar_error(F, skeleton_ref, skeleton_target):

    error = 0.0

    valid = 0

    for p_ref, p_dst in zip(skeleton_ref, skeleton_target):

        if np.any(np.isnan(p_ref)):
            continue

        if np.any(np.isnan(p_dst)):
            continue

        line = epipolar_line(F, p_ref)

        error += point_line_distance(line, p_dst)

        valid += 1

    if valid == 0:
        return np.inf

    return error / valid

def projection_matrix(cal):
    """
    Return projection matrix

        P = K [R|t]

    Parameters
    ----------
    cal : dict
        Calibration loaded per load_calibration()

    Returns
    -------
    P : ndarray (3x4)
    """

    K = cal["K"]

    E = cal["E"]

    P = K @ E[:3, :]

    return P


def match_skeletons(F, skeletons_src, skeletons_dst):
    """
    Associetes skeletons between two cameras using epipolar geometry.

    Parameters
    ----------
    F : ndarray (3x3)
        Fundamental matrix.

    skeletons_src : ndarray
        (N,17,2)

    skeletons_dst : ndarray
        (M,17,2)

    Returns
    -------
    cost : ndarray
        Matrix (N,M) containing the mean epipolar error.
    """

    N = skeletons_src.shape[0]
    M = skeletons_dst.shape[0]

    cost = np.zeros((N, M))

    for i in range(N):
        for j in range(M):

            cost[i, j] = skeleton_epipolar_error(
                F,
                skeletons_src[i],
                skeletons_dst[j]
            )

    return cost


def best_matches(F, skeletons_src, skeletons_dst):
    """
    Return the best match between skeletons.

    Returns
    -------
    matches : list
        [(idx_src, idx_dst, erro), ...]
    """

    cost = match_skeletons(
        F,
        skeletons_src,
        skeletons_dst
    )

    matches = []

    used_dst = set()

    for i in range(cost.shape[0]):

        order = np.argsort(cost[i])

        for j in order:

            if j not in used_dst:

                matches.append((i, j, cost[i, j]))
                used_dst.add(j)
                break

    return matches, cost
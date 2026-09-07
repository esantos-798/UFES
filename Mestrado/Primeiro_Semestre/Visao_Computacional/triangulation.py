import numpy as np
from epipolar_geometry import projection_matrix


def triangulate_point(P_list, points2d):
    """
    Triangules a point using multiple cameras.

    Parameters
    ----------
    P_list : list
        Projection matrix list (3x4)

    points2d : list
        Point List (x,y)

    Returns
    -------
    X : ndarray (3,)
        3D Coordinates
    """

    A = []

    for P, pt in zip(P_list, points2d):

        x, y = pt[:2]

        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)

    X = Vt[-1]

    X /= X[3]

    return X[:3]


def triangulate_skeleton(calibrations, all_skeletons, indices_per_cam):
    """
    indices_per_cam : dict {cam_id: skeleton_idx}
        Skeleton indice related to the same person in each camera.
    """
    P_list = [projection_matrix(cal) for cal in calibrations]

    skeleton3d = []
    for joint in range(17):
        points, Ps = [], []
        for cam, idx in indices_per_cam.items():
            pt = all_skeletons[cam][idx][joint]
            if np.any(np.isnan(pt)):
                continue
            points.append(pt)
            Ps.append(P_list[cam])
        X = triangulate_point(Ps, points)
        skeleton3d.append(X)

    return np.array(skeleton3d)

def project_point(P, X):
    """
    Projects a 3D point for a camera.

    Parameters
    ----------
    P : (3x4)
        Projection matrix.

    X : (3,)
        3D point.

    Returns
    -------
    (2,)
        Pixel coordinates.
    """

    Xh = np.append(X, 1)

    x = P @ Xh

    x /= x[2]

    return x[:2]

def reprojection_error(P, X, observation):
    """
    Distance between the observated point and reprojected point.
    """

    proj = project_point(P, X)

    return np.linalg.norm(proj - observation)
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import cv2

def project_no_distort(X, rvec, t, K):
    R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
    # print("X.shape=", X.shape) # (160, 3)
    # row 방식: Xc = Xw @ R.T + t, col 방식: Xc = R @ Xw + t
    XT = X @ R.T + t # 월드 -> 카메라 좌표 변환
    # row 방식: x = K.T @ X, col 방식: x = K @ X
    xT = XT @ K.T # 카메라 좌표 -> 픽셀 좌표
    xT = xT / xT[:,-1].reshape((-1, 1)) # 동차좌표 정규화

    return xT[:, 0:2]

def reproject_error_pnp(unknown, X, x, K):
    rvec, tvec = unknown[:3], unknown[3:]
    xp = project_no_distort(X, rvec, tvec, K)
    err = x - xp
    return err.ravel()

def solvePnP(obj_pts, img_pts, K):
    unknown_init = np.array([0, 0, 0, 0, 0, 1.]) # rvec(3), tvec(3)
    result = least_squares(reproject_error_pnp, unknown_init, args=(obj_pts, img_pts, K))

    return result['success'], result['x'][:3], result['x'][3:]

if __name__ == '__main__':
    f, cx, cy = 1000., 320., 240.
    obj_pts = np.loadtxt('../data/box.xyz')
    img_pts = np.loadtxt('../data/image_formation1.xyz')[:,:2].copy()
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    dist_coeff = np.zeros(4)

    _, rvec, tvec = solvePnP(obj_pts, img_pts, K)
    R = Rotation.from_rotvec(rvec.flatten()).as_matrix()
    my_ori = Rotation.from_matrix(R.T).as_euler('xyz')
    my_pos = -R.T @ tvec

    _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeff)
    R = Rotation.from_rotvec(rvec.flatten()).as_matrix() # 월드 -> 카메라 회전 행렬
    cv_ori = Rotation.from_matrix(R.T).as_euler('xyz') # R.T : 카메라 -> 월드 회전
    cv_pos = -R.T @ tvec.flatten() # C = -R^T @ t => 카메라의 월드 좌표 위치

    print('\n### Ground Truth')
    print('* Camera orientation: [-15, 15, 0] [deg]')
    print('* Camera position   : [-2, -2, 0] [m]')
    print('\n### My Camera Pose')
    print(f'* Camera orientation: {np.rad2deg(my_ori)} [deg]')
    print(f'* Camera position   : {my_pos} [m]')
    print('\n### OpenCV Camera Pose')
    print(f'* Camera orientation: {np.rad2deg(cv_ori)} [deg]')
    print(f'* Camera position   : {cv_pos} [m]')

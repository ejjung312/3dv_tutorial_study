import numpy as np
import cv2

def triangulatePoints(P0, P1, pts0, pts1):
    Xs = []
    for (p, q) in zip(pts0.T, pts1.T):
        # w=1로 둠
        A = np.vstack((p[0] * P0[2] - P0[0],
                       p[1] * P0[2] - P0[1],
                       q[0] * P1[2] - P1[0],
                       q[1] * P1[2] - P1[1]))
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        Xs.append(Vt[-1])
        
    return np.vstack(Xs).T

if __name__ == '__main__':
    f, cx, cy = 1000., 320., 240.
    pts0 = np.loadtxt('../data/image_formation0.xyz')[:,:2]
    pts1 = np.loadtxt('../data/image_formation1.xyz')[:,:2]
    output_file = 'triangulation_implement.xyz'

    F, _ = cv2.findFundamentalMat(pts0, pts1, cv2.FM_8POINT)
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    E = K.T @ F @ K
    _, R, t, _ = cv2.recoverPose(E, pts0, pts1)

    # np.eye(3, 4) : 회전 R = I, 이동 t = 0 => 카메라 1을 월드 원점으로 둠
    # Projection matrix P : 3D 월드 좌표를 2D 이미지 좌표로 바꾸기 위해 필요한(사상하는) 3x4 행렬
    P0 = K @ np.eye(3, 4, dtype=np.float32) # x = PX = K[R|t]X, P = K[R|t]
    Rt = np.hstack((R, t)) # (3, 4)
    P1 = K @ Rt # # 카메라 2의 Projection matrix
    X = triangulatePoints(P0, P1, pts0.T, pts1.T)
    # print("X.shape=", X.shape) # (4, 160)
    # print("X=",X)
    X /= X[3]
    X = X.T # [X,Y,Z,1] 형태
    # print("X.shape=", X.shape) # (160, 4)

    np.savetxt(output_file, X)
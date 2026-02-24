import numpy as np
import cv2

def findFundamentalMat(pts1, pts2):
    if len(pts1) == len(pts2):
        if pts1.shape[1] == 2:
            pts1 = np.hstack((pts1, np.ones((len(pts1), 1), dtype=pts1.dtype)))
        if pts2.shape[1] == 2:
            pts2 = np.hstack((pts2, np.ones((len(pts2), 1), dtype=pts2.dtype)))
        
        A = []
        for p, q in zip(pts1, pts2):
            A.append([q[0]*p[0], q[0]*p[1], q[0]*p[2], q[1]*p[0], q[1]*p[1], q[1]*p[2], q[2]*p[0], q[2]*p[1], q[2]*p[2]])
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        x = Vt[-1]

        # rank 2 강제
        # print("x.shape=", x.shape) # (9, 1)
        F = x.reshape(3, -1)
        # print("F.shape", F.shape) # (3,3)
        U, S, Vt = np.linalg.svd(F)
        S[-1] = 0 # [σ1, σ2, 0] 형태
        F = U @ np.diag(S) @ Vt # np.diag 벡터를 대각행렬로 만듬
        """
        [
            [σ1, 0, 0],
            [0, σ2, 0],
            [0, 0, 0] # σ3을 0으로 바꿈(S[-1] = 0)
        ]
        """
        
        return F / F[-1,-1] # 마지막 원소로 정규화

if __name__ == "__main__":
    pts0 = np.loadtxt('../data/image_formation0.xyz')
    pts1 = np.loadtxt('../data/image_formation1.xyz')

    # print("pts0.shape=", pts0.shape) # (160, 3)
    # print("pts1.shape=", pts1.shape) # (160, 3)

    my_F = findFundamentalMat(pts0, pts1)
    cv_F, _ = cv2.findFundamentalMat(pts0, pts1, cv2.FM_8POINT)

    print('\n### My Fundamental Matrix')
    print(my_F)
    print('\n### OpenCV Fundamental Matrix')
    print(cv_F)
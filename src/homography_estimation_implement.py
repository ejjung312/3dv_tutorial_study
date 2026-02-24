import numpy as np
import cv2

def getPerspectiveTransform(src, dst):
    if len(src) == len(dst):
        if src.shape[1] == 2:
            src = np.hstack((src, np.ones((len(src), 1), dtype=src.dtype)))
        if dst.shape[1] == 2:
            dst = np.hstack((dst, np.ones((len(dst), 1), dtype=dst.dtype)))
        # print("src=", src)
        # print("dst=", dst)
        """
        [
            [115, 401, 1], 
            [776, 180, 1], 
            [330, 793, 1], 
            [1080, 383, 1]
        ]
        [
            [0, 0, 1], 
            [900, 0, 1], 
            [0, 500, 1], 
            [900, 500, 1]
        ]
        """
        A = []
        for p, q in zip(src, dst):
            A.append([0, 0, 0, q[2]*p[0], q[2]*p[1], q[2]*p[2], -q[1]*p[0], -q[1]*p[1], -q[1]*p[2]])
            A.append([q[2]*p[0], q[2]*p[1], q[2]*p[2], 0, 0, 0, -q[0]*p[0], -q[0]*p[1], -q[0]*p[2]])
        
        # print("A=", A)
        _, _, Vt = np.linalg.svd(A)
        x = Vt[-1]
        # print("x=", x)
        """
        x= [ 1.26858125e-03 -6.95777980e-04  1.33120127e-01  7.54045501e-04 2.25531256e-03 -9.91095570e-01 -1.24497338e-07  1.52418304e-06 9.24811586e-04]
        """
        H = x.reshape(3, -1) / x[-1]

        return H

if __name__ == "__main__":
    src = np.array([[115, 401], [776, 180], [330, 793], [1080, 383]], dtype=np.float32)
    dst = np.array([[0, 0], [900, 0], [0, 500], [900, 500]], dtype=np.float32)

    # 평면 위의 2D 점 → 다른 평면 위의 2D 점으로 보내는 호모그래피(Homography) 변환
    my_H = getPerspectiveTransform(src, dst)
    cv_H = cv2.getPerspectiveTransform(src, dst)

    print('\n### My Planar Homography')
    print(my_H)
    print('\n### OpenCV Planar Homography')
    print(cv_H)
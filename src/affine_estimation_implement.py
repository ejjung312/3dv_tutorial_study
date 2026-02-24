import numpy as np
import cv2

def getAffineTransform(src, dst):
    if len(src) == len(dst):
        A, b = [], []
        for p, q in zip(src, dst):
            A.append([p[0], p[1], 0, 0, 1, 0])
            A.append([0, 0, p[0], p[1], 0, 1])
            b.append(q[0])
            b.append(q[1])
        # print("A=", A)
        # print("b=", b)
        x = np.linalg.pinv(A) @ b # x = A^-1 @ b

        """
        a11 a12 tx
        a21 a22 ty
        """
        H = np.array([[x[0], x[1], x[4]], [x[2], x[3], x[5]]])
        return H

if __name__ == "__main__":
    src = np.array([[115, 401], 
                    [776, 180], 
                    [330, 793]], dtype=np.float32)
    dst = np.array([[0, 0], [900, 0], [0, 500]], dtype=np.float32)

    my_H = getAffineTransform(src, dst)
    cv_H = cv2.getAffineTransform(src, dst)

    print('\n### My Affine Transformation')
    print(my_H)
    print('\n### OpenCV Affine Transformation')
    print(cv_H)
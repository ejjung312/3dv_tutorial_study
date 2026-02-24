import numpy as np
import matplotlib.pyplot as plt
import cv2

f, cx, cy = 1000., 320., 240.
pts0 = np.loadtxt('../data/image_formation0.xyz')[:,:2]
pts1 = np.loadtxt('../data/image_formation1.xyz')[:,:2]
output_file = 'triangulation.xyz'

F, _ = cv2.findFundamentalMat(pts0, pts1, cv2.FM_8POINT) # 8-point algorithm
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])
E = K.T @ F @ K # E = K^T @ F @ K
_, R, t, _ = cv2.recoverPose(E, pts0, pts1) # Z>0 조건 만족 R, t

# np.eye(3, 4) : 회전 R = I, 이동 t = 0 => 카메라 1을 월드 원점으로 둠
# Projection matrix P : 3D 월드 좌표를 2D 이미지 좌표로 바꾸기 위해 필요한(사상하는) 3x4 행렬
P0 = K @ np.eye(3, 4, dtype=np.float32) # x = PX = K[R|t]X, P = K[R|t]
Rt = np.hstack((R, t))
# print("Rt.shape=", Rt.shape) # (3, 4)
P1 = K @ Rt # 카메라 2의 Projection matrix
X = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T) # (4, N) 동차 좌표 반환 X = [X, Y, Z, W]^T
X /= X[3] # 동차좌표 -> 유클리드 좌표 변환 (X/W, Y/W, Z/W, 1)
X = X.T

np.savetxt(output_file, X)

ax = plt.figure(layout='tight').add_subplot(projection='3d')
ax.plot(X[:,0], X[:,1], X[:,2], 'ro') # X[:,0] x좌표, X[:,1] y좌표, X[:,2] z좌표 리스트
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.grid(True)
plt.show()
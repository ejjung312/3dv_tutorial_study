import numpy as np
import cv2 
from scipy.spatial.transform import Rotation

f, cx, cy, noise_std = 1000, 320, 240, 1
img_res = (640, 480)
cam_pos = [[0, 0, 0], [-2, -2, 0], [2, 2, 0], [-2, 2, 0], [2, -2, 0]]
cam_ori = [[0, 0, 0], [-15 , 15, 0], [15, -15, 0], [15, 15, 0], [-15, -15, 0]]

X = np.loadtxt('../data/box.xyz')

K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])

for i, (pos, ori) in enumerate(zip(cam_pos, cam_ori)):
    Rc = Rotation.from_euler('zyx', ori[::-1], degrees=True).as_matrix()
    R = Rc.T # R = Rc^T
    t = -Rc.T @ pos # t = -Rc^T @ tc

    x = K @ (R @ X.T + t.reshape(-1, 1)) # x = K(RX + t)
    x /= x[-1] # x = [x/w, y/w, w/w]^T

    # 실제 카메라가 픽셀을 측정할 때 오차가 존재하기 때문에 노이즈 추가
    noise = np.random.normal(scale=noise_std, size=(2, len(X)))
    x[0:2,:] += noise

    img = np.zeros(img_res[::-1], dtype=np.uint8)
    for c in range(x.shape[1]):
        cv2.circle(img, x[0:2,c].astype(np.int32), 2, 255, -1)
    
    cv2.imshow(f'Image Formation {i}', img)
    np.savetxt(f'image_formation{i}.xyz', x.T) # Size: N x 2

cv2.waitKey()
cv2.destroyAllWindows()
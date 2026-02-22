import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def mouse_event_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['dragged'] = True
        param['xy_s'] = (x, y)
        param['xy_e'] = (0, 0)
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['dragged']:
            param['xy_e'] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if param['dragged']:
            param['dragged'] = False
            param['xy_e'] = (x, y)

if __name__ == '__main__':
    img_file = '../data/daejeon_station.png'

    f, cx, cy, L = 810.5, 480, 270, 3.31 # px px px m

    cam_ori = [-18.7, -8.2, 2.0] # 월드 좌표계 기준 카메라 좌표계가 얼마나 회전했는지 3개 각도
    grid_x, grid_z = (-2,3), (5,36)

    img = cv2.imread(img_file)

    mouse_state = {'dragged': False, 'xy_s': (0, 0), 'xy_e': (0, 0)}
    cv2.namedWindow('Object Localization and Measurement')
    cv2.setMouseCallback('Object Localization and Measurement', mouse_event_handler, mouse_state)

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    Rc = Rotation.from_euler('zyx', cam_ori[::-1], degrees=True).as_matrix() # 월드 좌표계 기준 카메라 회전
    tc = np.array([0, -L, 0]) # 월드 좌표계 기준 카메라 위치
    R = Rc.T # 카메라 좌표계 기준 월드 좌표계 회전
    t = -Rc.T @ tc.T # 카메라 좌표계 기존 월드 좌표계 위치

    # Extrinsic + Intrinsic을 직접 검증하는 코드
    # x 방향 격자선
    for x in range(*grid_x):
        s, e = [x, 0, grid_z[0]], [x, 0, grid_z[1] - 1]
        # 월드 -> 카메라 -> 이미지 투영
        p = K @ (R @ s + t) 
        q = K @ (R @ e + t)
        # x = [u,v,w]^T, u'=u/w, v'=v/w
        cv2.line(img, (int(p[0] / p[2]), int(p[1] / p[2])), (int(q[0] / q[2]), int(q[1] / q[2])), (64, 128, 64), 1)
    
    # z 방향 격자선
    for z in range(*grid_z):
        s, e = [grid_x[0], 0, z], [grid_x[1] - 1, 0, z]
        p = K @ (R @ s + t)
        q = K @ (R @ e + t)
        cv2.line(img, (int(p[0] / p[2]), int(p[1] / p[2])), (int(q[0] / q[2]), int(q[1] / q[2])), (64, 128, 64), 1)

    while True:
        img_copy = img.copy()

        if mouse_state['xy_e'][0] > 0 and mouse_state['xy_e'][1] > 0:
            # 위치와 높이 계산
            xs, ys = mouse_state['xy_s']
            xe, ye = mouse_state['xy_e']
            # 02 - 29p
            c = R.T @ [xs - cx, ys - cy, f]
            h = R.T @ [xe - cx, ye - cy, f]

            if c[1] < 1e-6:
                continue
            
            # 02 - 29p
            X = c[0] / c[1] * L
            Z = c[2] / c[1] * L
            H = (c[1]/c[2] - h[1]/h[2]) * Z

            cv2.line(img_copy, mouse_state['xy_s'], mouse_state['xy_e'], (0,0,255), 2)
            cv2.circle(img_copy, mouse_state['xy_e'], 4, (255, 0, 0), -1)
            cv2.circle(img_copy, mouse_state['xy_s'], 4, (0, 255, 0), -1)
            info = f'X: {X:.3f}, Z: {Z:.3f}, H: {H:.3f}'
            cv2.putText(img_copy, info, np.array(mouse_state['xy_s']) + (-20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        
        cv2.imshow('Object Localization and Measurement', img_copy)
        key = cv2.waitKey(10)
        if key == 27: # ESC
            break

    cv2.destroyAllWindows()

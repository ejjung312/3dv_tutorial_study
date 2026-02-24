import numpy as np
import cv2
import random

def mouse_event_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))

def draw_straight_line(img, line, color, thickness=1):
    assert img.ndim >= 2
    h, w, *_ = img.shape
    a, b, c = line # Line: ax + by + c = 0
    # a ≈ 0 인데 x로 나누거나, b ≈ 0 인데 y로 나누면 -> 폭발
    # 따라서 절댓값이 더 큰 계수를 기준으로 계산
    if abs(a) > abs(b): # x 기준 계산
        pt1 = (int(c / -a), 0) # y=0일 때
        pt2 = (int((b*h + c) / -a), h) # y=h일 때
    else: # y 기준 계산
        pt1 = (0, int(c / -b)) # x=0일 때
        pt2 = (w, int((a*w + c) / -b)) # x=w 일 때
    cv2.line(img, pt1, pt2, color, thickness)

if __name__ == "__main__":
    img1 = cv2.imread('../data/KITTI07/image_0/000000.png', cv2.IMREAD_COLOR)
    img2 = cv2.imread('../data/KITTI07/image_0/000023.png', cv2.IMREAD_COLOR)
    assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'

    # Note) `F` is derived from `fundamental_mat_estimation.py`.
    F = np.array([[ 3.34638533e-07,  7.58547151e-06, -2.04147752e-03],
                  [-5.83765868e-06,  1.36498636e-06,  2.67566877e-04],
                  [ 1.45892349e-03, -4.37648316e-03,  1.00000000e+00]])
    
    wnd1_name, wnd2_name = 'Epipolar Line: Image #1', 'Epipolar Line: Image #2'
    img1_pts, img2_pts = [], []
    cv2.namedWindow(wnd1_name)
    cv2.namedWindow(wnd2_name)
    cv2.setMouseCallback(wnd1_name, mouse_event_handler, img1_pts)
    cv2.setMouseCallback(wnd2_name, mouse_event_handler, img2_pts)
    cv2.imshow(wnd1_name, img1)
    cv2.imshow(wnd2_name, img2)

    while True:
        if len(img1_pts) > 0:
            for x, y in img1_pts:
                color = (random.randrange(256), random.randrange(256), random.randrange(256))
                cv2.circle(img1, (x, y), 4, color, -1)
                epipolar_line = F @ [[x], [y], [1]] # l = Fx
                draw_straight_line(img2, epipolar_line, color, 2)
            img1_pts.clear()
        
        if len(img2_pts) > 0:
            for x, y in img2_pts:
                color = (random.randrange(256), random.randrange(256), random.randrange(256))
                cv2.circle(img2, (x, y), 4, color, -1)
                epipolar_line = F.T @ [[x], [y], [1]] # F^T는 2->1 방향 전환
                draw_straight_line(img1, epipolar_line, color, 2)
            img2_pts.clear()
        
        cv2.imshow(wnd2_name, img2)
        cv2.imshow(wnd1_name, img1)
        key = cv2.waitKey(10)
        if key == 27: # ESC
            break
    
    cv2.destroyAllWindows()
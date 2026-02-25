import numpy as np
import matplotlib.pyplot as plt
import cv2

video_file = '../data/KITTI07/image_0/%06d.png'
min_track_error = 5

video = cv2.VideoCapture(video_file)
assert video.isOpened()

_, gray_prev = video.read()
assert gray_prev.size > 0
if gray_prev.ndim >= 3 and gray_prev.shape[2] > 1:
    gray_prev = cv2.cvtColor(gray_prev, cv2.COLOR_BGR2GRAY)

while True:
    # Grab an image from the video
    valid, img = video.read()
    if not valid:
        break
    if img.ndim >= 3 and img.shape[2] > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 추적하기 좋은 코너 검출
    pts_prev = cv2.goodFeaturesToTrack(gray_prev, 2000, 0.01, 10)
    # 이전 프레임과 현재 프레임의 픽셀 이동량 (dx, dy)을 계산
    # error: 밝기 차이 기반 residual(intensity residual)
    pts, status, error = cv2.calcOpticalFlowPyrLK(gray_prev, gray, pts_prev, None)
    gray_prev = gray

    if img.ndim < 3 or img.shape[2] < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for pt, pt_prev, tracked, err in zip(pts, pts_prev, status, error):
        if tracked and err < min_track_error:
            cv2.line(img, pt_prev.flatten().astype(np.int32), pt.flatten().astype(np.int32), (0,255,0))
    
    cv2.imshow('KLT Feature Tracking', img)
    key = cv2.waitKey(30)
    if key == ord(' '):
        key = cv2.waitKey()
    if key == 27: # ESC
        break

video.release()
cv2.destroyAllWindows()
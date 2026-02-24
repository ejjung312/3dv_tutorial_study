import numpy as np
import cv2

video = cv2.VideoCapture("../data/traffic.avi")
assert video.isOpened()

_, gray_ref = video.read()
assert gray_ref.size > 0
if gray_ref.ndim >= 3:
    gray_ref = cv2.cvtColor(gray_ref, cv2.COLOR_BGR2GRAY)
# 추적하기 좋은 코너 검출
pts_ref = cv2.goodFeaturesToTrack(gray_ref, 2000, 0.01, 10)

while True:
    valid, img = video.read()
    if not valid:
        break
    if img.ndim >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # optical flow 계산
    # optical flow: 같은 점은 밝기가 거의 변하지 않는다고 가정하고 이전 프레임과 현재 프레임의 픽셀 이동량 (dx, dy)을 계산
    pts, status, error = cv2.calcOpticalFlowPyrLK(gray_ref, gray, pts_ref, None)
    # 이전 프레임과 현재 프레임의 호모그래피 계산
    H, inlier_mask = cv2.findHomography(pts, pts_ref, cv2.RANSAC)

    warp = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))

    for pt, pt_ref, inlier in zip(pts, pts_ref, inlier_mask):
        color = (0,0,255) if inlier else (0,127,0)
        cv2.line(img, pt.flatten().astype(np.int32), pt_ref.flatten().astype(np.int32), color)
    
    cv2.imshow('2D Video Stabilization', np.hstack((img, warp)))
    key = cv2.waitKey(30)
    if key == ord(' '):
        key = cv2.waitKey()
    if key == 27: # ESC
        break

video.release()
cv2.destroyAllWindows()
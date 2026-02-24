import numpy as np
import matplotlib.pyplot as plt
import cv2

video_file = '../data/KITTI07/image_0/%06d.png'
f, cx, cy = 707.0912, 601.8873, 183.1104
use_5pt = True
min_inlier_num = 100
min_inlier_ratio = 0.2
traj_file = 'vo_epipolar.xyz'

video = cv2.VideoCapture(video_file)
assert video.isOpened()

_, gray_prev = video.read()
assert gray_prev.size > 0
if gray_prev.ndim >= 3 and gray_prev.shape[2] > 1:
    gray_prev = cv2.cvtColor(gray_prev, cv2.COLOR_BGR2GRAY)

plt.ion()
traj_axes = plt.figure(layout='tight').add_subplot(projection='3d')
traj_axes.set_xlabel('X [m]')
traj_axes.set_ylabel('Y [m]')
traj_axes.set_zlabel('Z [m]')
traj_axes.grid(True)
traj_axes.view_init(azim=-90)
# 언패킹 - 리스트 안의 첫 번째 원소를 traj_line에 넣음
traj_line, = plt.plot([], [], [], 'b-') # 'b-': 파란색 실선

K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])
camera_pose = np.eye(4) # 카메라 전역 pose
# print("camera_pose.shape=", camera_pose.shape) (4, 4)
camera_traj = np.zeros((1, 3))

while True:
    valid, img = video.read()
    if not valid:
        break
    if img.ndim >= 3 and img.shape[2] > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 추적하기 좋은 코너 검출
    pts_prev = cv2.goodFeaturesToTrack(gray_prev, 2000, 0.01, 10)
    # optical flow 계산
    # optical flow: 같은 점은 밝기가 거의 변하지 않는다고 가정하고 이전 프레임과 현재 프레임의 픽셀 이동량 (dx, dy)을 계산
    pts, status, error = cv2.calcOpticalFlowPyrLK(gray_prev, gray, pts_prev, None)
    gray_prev = gray

    if use_5pt:
        E, inlier_mask = cv2.findEssentialMat(pts_prev, pts, f, (cx, cy), cv2.FM_RANSAC, 0.99, 1)
    else:
        F, inlier_mask = cv2.findFundamentalMat(pts_prev, pts, cv2.FM_RANSAC, 1, 0.99)

    # 카메라의 앞쪽(촬영 방향, Z>0)을 보는 R, t
    # 카메라1 기준으로 카메라2의 움직임(R, t)
    inlier_num, R, t, inlier_mask = cv2.recoverPose(E, pts_prev, pts, focal=f, pp=(cx, cy), mask=inlier_mask)
    inlier_ratio = inlier_num / len(pts)

    info_color = (0, 255, 0)
    if inlier_num > min_inlier_num and inlier_ratio > min_inlier_ratio:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        # 카메라 2(다음 프레임)의 월드 위치
        camera_pose = camera_pose @ np.linalg.inv(T) # @: 행렬 곱 / np.linalg.inv: 역행렬
        info_color = (0,0,255)
    
    x, y, z = camera_pose[:3, 3] # 3x1 translation 벡터 (tx, ty, tz)
    camera_traj = np.vstack((camera_traj, [x, y, z])) # row 추가
    traj_axes.set_xlim(min(camera_traj[:,0]), max(camera_traj[:,0])) # 축 범위 갱신
    traj_axes.set_ylim(min(camera_traj[:,1]), max(camera_traj[:,1]))
    traj_axes.set_zlim(min(camera_traj[:,2]), max(camera_traj[:,2]))
    traj_axes.set_aspect('equal') # X, Y, Z 축 스케일을 동일하게 유지
    traj_line.set_data_3d(camera_traj[:,0], camera_traj[:,1], camera_traj[:,2]) # 라인 업데이트
    plt.draw()

    if img.ndim < 3 or img.shape[2] < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pt, pt_prev, inlier in zip(pts, pts_prev, inlier_mask):
        color = (0, 0, 255) if inlier else (0, 127, 0)
        cv2.line(img, pt_prev.flatten().astype(np.int32), pt.flatten().astype(np.int32), color)
    
    info = f'Inliers: {inlier_num} ({inlier_ratio*100:.0f}%), XYZ: [{x:.3f} {y:.3f} {z:.3f}]'
    cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, info_color)
    cv2.imshow('Monocular Visual Odometry (Epipolar)', img)
    key = cv2.waitKey(30)
    if key == ord(' '):
        key = cv2.waitKey()
    if key == 27: # ESC
        break

np.savetxt(traj_file, camera_traj)
video.release()
cv2.destroyAllWindows()
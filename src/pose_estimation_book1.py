"""
내부 파라미터 K가 주어졌을 때,
각 이미지의 외부 파라미터(R, t) 추정
-> 카메라가 이미 캘리브레이션 되어 있음
"""
import numpy as np
import cv2

video_file, cover_file = '../data/blais.mp4', '../data/blais.jpg'
f, cx, cy = 1000, 320, 240
min_inlier_num = 100

fdetector = cv2.ORB_create()
fmatcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')

obj_image = cv2.imread(cover_file)
assert obj_image is not None

obj_keypoints, obj_descriptors = fdetector.detectAndCompute(obj_image, None)
assert len(obj_keypoints) >= min_inlier_num
fmatcher.add(obj_descriptors)

video = cv2.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given video, ' + video_file

box_lower = np.array([[30, 145, 0], [30, 200, 0], [200, 200, 0], [200, 145, 0]], dtype=np.float32)
box_upper = np.array([[30, 145, -50], [30, 200, -50], [200, 200, -50], [200, 145, -50]], dtype=np.float32)

K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]], dtype=np.float32)
dist_coeff = np.zeros(5) # [0. 0. 0. 0. 0.]

while True:
    valid, img = video.read()
    if not valid:
        break

    img_keypoints, img_descriptors = fdetector.detectAndCompute(img, None)
    match = fmatcher.match(img_descriptors, obj_descriptors)
    if len(match) < min_inlier_num:
        continue

    obj_pts, img_pts = [], []
    for m in match:
        obj_pts.append(obj_keypoints[m.trainIdx].pt)
        img_pts.append(img_keypoints[m.queryIdx].pt)
    obj_pts = np.array(obj_pts, dtype=np.float32)
    # print("obj_pts.shape=", obj_pts.shape) # (500, 2)
    obj_pts = np.hstack((obj_pts, np.zeros((len(obj_pts), 1), dtype=np.float32))) # Z=0 추가
    # print("obj_pts.shape=", obj_pts.shape) # (500, 3)
    img_pts = np.array(img_pts, dtype=np.float32)

    # PnP(외부 파라미터 추정)를 RANSAC 기반으로 수행하여 이상치(outlier)를 제거하면서 pose를 추정
    ret, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, dist_coeff, useExtrinsicGuess=False, iterationsCount=500, reprojectionError=2.0, confidence=0.99)

    inlier_mask = np.zeros(len(match), dtype=np.uint8)
    inlier_mask[inliers] = 1
    img_result = cv2.drawMatches(img, img_keypoints, obj_image, obj_keypoints, match, None, (0,0,255), (0,127,0), inlier_mask)

    inlier_num = sum(inlier_mask)
    if inlier_num > min_inlier_num:
        # PnP로 pose(외부 파라미터) 추정
        ret, rvec, tvec = cv2.solvePnP(obj_pts[inliers], img_pts[inliers], K, dist_coeff)

        # 3D 점을 2D로 투영
        line_lower, _ = cv2.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv2.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        
        cv2.polylines(img_result, [np.int32(line_lower)], True, (255,0,0), 2)
        cv2.polylines(img_result, [np.int32(line_upper)], True, (0,0,255), 2)
        for b, t in zip(line_lower, line_upper):
            cv2.line(img_result, np.int32(b.flatten()), np.int32(t.flatten()), (0,255,0), 2)
        
        ratio = (inlier_num/len(match)) * 100
        info = f'Inliers: {inlier_num} ({ratio}%), Focal length: {f}'
        cv2.putText(img_result, info, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv2.imshow('Pose Estimation (Book)', img_result)
    key = cv2.waitKey(1)
    if key == ord(' '):
        key = cv2.waitKey()
    if key == 27: # ESC
        break

video.release()
cv2.destroyAllWindows()
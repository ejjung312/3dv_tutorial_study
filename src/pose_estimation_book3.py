"""
호모그래피 H 추정 후
내부 파라미터 K 추정
"""
import numpy as np
import cv2

video_file, cover_file = '../data/blais.mp4', '../data/blais.jpg'
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

calib_param = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY

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

    # 인라이어 찾기
    H, inlier_mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, 2)
    inlier_mask = inlier_mask.flatten()
    img_result = cv2.drawMatches(img, img_keypoints, obj_image, obj_keypoints, match, None, (0,0,255), (0,127,0), inlier_mask)

    inlier_num = sum(inlier_mask)
    if inlier_num > min_inlier_num:
        # 인라이어만으로 카메라 캘리브레이션
        # rvecs[i], tvecs[i] = i번째 이미지에 대한 외부 파라미터
        ret, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera([obj_pts[inlier_mask.astype(bool)]], [img_pts[inlier_mask.astype(bool)]], (img.shape[1], img.shape[0]), None, None, None, None, calib_param)
        # print("[obj_pts[inliers]]=", [obj_pts[inliers]])
        # print("[img_pts[inliers]]=", [img_pts[inliers]])
        rvec, tvec = rvecs[0], tvecs[0] # [obj_pts[inliers]], [img_pts[inliers]] 길이가 1이기 때문에 [0] 접근

        # 3D 점을 2D로 투영
        line_lower, _ = cv2.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv2.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
        
        cv2.polylines(img_result, [np.int32(line_lower)], True, (255,0,0), 2)
        cv2.polylines(img_result, [np.int32(line_upper)], True, (0,0,255), 2)
        for b, t in zip(line_lower, line_upper):
            cv2.line(img_result, np.int32(b.flatten()), np.int32(t.flatten()), (0,255,0), 2)
        
        ratio = (inlier_num/len(match)) * 100
        info = f'Inliers: {inlier_num} ({ratio}%), Focal length: {K[0, 0]}'
        cv2.putText(img_result, info, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    cv2.imshow('Pose Estimation (Book)', img_result)
    key = cv2.waitKey(1)
    if key == ord(' '):
        key = cv2.waitKey()
    if key == 27: # ESC
        break

video.release()
cv2.destroyAllWindows()
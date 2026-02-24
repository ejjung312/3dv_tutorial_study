import numpy as np
import cv2

# Load two images
img1 = cv2.imread('../data/KITTI07/image_0/000000.png')
img2 = cv2.imread('../data/KITTI07/image_0/000023.png')
assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'
f, cx, cy = 707.0912, 601.8873, 183.1104
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]])

fdetector = cv2.BRISK_create()
keypoints1, descriptors1 = fdetector.detectAndCompute(img1, None)
keypoints2, descriptors2 = fdetector.detectAndCompute(img2, None)

fmatcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
match = fmatcher.match(descriptors1, descriptors2)

pts1, pts2 = [], []
for i in range(len(match)):
    pts1.append(keypoints1[match[i].queryIdx].pt)
    pts2.append(keypoints2[match[i].trainIdx].pt)
pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)
F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.5, 0.999)
print(f'* F = {F}')
print(f'* The number of inliers = {sum(inlier_mask.ravel())}')

E = K.T @ F @ K
# E에서 R, t를 복원
# positive_num: 3D 점이 해당 카메라의 앞쪽(촬영 방향, Z>0)에 존재하는 개수
# positive_mask: inlier_mask에서 카메라 앞쪽에 있는 점
positive_num, R, t, positive_mask = cv2.recoverPose(E, pts1, pts2, K, mask=inlier_mask)
print(f'* R = {R}')
print(f'* t = {t}')
# Xc = RXw + t
print(f'* The position of Image #2 = {-R.T @ t}') # 카메라 1 좌표계를 월드로 정의했을 때 카메라 2의 월드 좌표계 위치(카메라 중심)
print(f'* The number of positive-depth inliers = {sum(positive_mask.ravel())}')

img_matched = cv2.drawMatches(img1, keypoints1, img2, keypoints2, match, None, None, None, matchesMask=inlier_mask.ravel().tolist())

cv2.namedWindow('Fundamental Matrix Estimation', cv2.WINDOW_NORMAL)
cv2.imshow('Fundamental Matrix Estimation', img_matched)
cv2.waitKey(0)
cv2.destroyAllWindows()

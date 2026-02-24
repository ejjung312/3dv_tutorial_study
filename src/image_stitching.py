import numpy as np
import cv2

img1 = cv2.imread("../data/hill01.jpg")
img2 = cv2.imread("../data/hill02.jpg")
assert (img1 is not None) and (img2 is not None), "Cannot read the given images"

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

H, inlier_mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
img_merged = cv2.warpPerspective(img2, H, (img1.shape[1]*2, img1.shape[0]))
img_merged[:,:img1.shape[1]] = img1

img_matched = cv2.drawMatches(img1, keypoints1, img2, keypoints2, match, None, None, None, matchesMask=inlier_mask.ravel().tolist())

# merge = np.vstack((np.hstack((img1, img2)), img_matched, img_merged))
merge = np.vstack((img_matched, img_merged))

cv2.imshow('Planar Image Stitching', merge)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import cv2
import time

img1 = cv2.imread('../data/hill01.jpg')
img2 = cv2.imread('../data/hill02.jpg')
assert (img1 is not None) and (img2 is not None), 'Cannot read the given images'

features = [
    {'name': 'AKAZE',   'detector': cv2.AKAZE_create(),               'matcher': cv2.DescriptorMatcher_create('BruteForce-Hamming')},
    {'name': 'BRISK',   'detector': cv2.BRISK_create(),               'matcher': cv2.DescriptorMatcher_create('BruteForce-Hamming')},
    {'name': 'FAST',    'detector': cv2.FastFeatureDetector_create(), 'matcher': None}, # No descriptor
    {'name': 'GFTT',    'detector': cv2.GFTTDetector_create(),        'matcher': None}, # No descriptor
    {'name': 'KAZE',    'detector': cv2.KAZE_create(),                'matcher': None}, # No descriptor
    {'name': 'MSER',    'detector': cv2.MSER_create(),                'matcher': None}, # No descriptor
    {'name': 'ORB',     'detector': cv2.ORB_create(),                 'matcher': cv2.DescriptorMatcher_create('BruteForce-Hamming')},
    {'name': 'SIFT',    'detector': cv2.SIFT_create(),                'matcher': cv2.DescriptorMatcher_create('BruteForce')},
]

f_select = 7

while True:
    time_start = time.time()
    keypoints1 = features[f_select]['detector'].detect(img1)
    keypoints2 = features[f_select]['detector'].detect(img2)
    time_detect = time.time()

    if features[f_select]['matcher'] is not None:
        keypoints1, descriptors1 = features[f_select]['detector'].compute(img1, keypoints1)
        keypoints2, descriptors2 = features[f_select]['detector'].compute(img2, keypoints2)
        time_compute = time.time()

        match = features[f_select]['matcher'].match(descriptors1, descriptors2)
        time_match = time.time()
    else:
        time_compute = time_detect
        time_match = time_compute

    if features[f_select]['matcher'] is not None:
        img_merged = cv2.drawMatches(img1, keypoints1, img2, keypoints2, match, None)
    else:
        img1_keypts = cv2.drawKeypoints(img1, keypoints1, None)
        img2_keypts = cv2.drawKeypoints(img2, keypoints2, None)
        img_merged = np.hstack((img1_keypts, img2_keypts))
    
    info = features[f_select]['name'] + f': ({(time_detect-time_start)*1000:.0f} + {(time_compute-time_detect)*1000:.0f} + {(time_match-time_compute)*1000:.0f}) = {(time_match-time_start)*1000:.0f} [msec]'
    cv2.putText(img_merged, info, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    cv2.imshow('Feature Matching', img_merged)

    key = cv2.waitKey(0)
    if key == 27: # ESC
        break
    elif key == ord('-') or key == ord('_'):
        f_select = (f_select - 1) % len(features)
    else:
        f_select = (f_select + 1) % len(features)

cv2.destroyAllWindows()
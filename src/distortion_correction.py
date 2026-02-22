import numpy as np
import cv2

video_file = '../data/chessboard.avi'
K = np.array([[432.7390364738057, 0, 476.0614994349778],
              [0, 431.2395555913084, 288.7602152621297],
              [0, 0, 1]])
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])

video = cv2.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

show_rectify = True
map1, map2 = None, None
while True:
    valid, img = video.read()
    if not valid:
        break

    info = "Original"
    if show_rectify:
        if map1 is None or map2 is None:
            # 이미지 보정을 위한 픽셀 매핑 테이블을 계산하는 함수
            map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv2.CV_32FC1)
            # print("map1=", map1, "shape=", map1.shape)
            # print("map2=", map2, "shape=", map2.shape)
        
        # map을 적용해서 이미지 변환
        img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        info = "Rectified"
    cv2.putText(img, info, (10,25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0))

    cv2.imshow("Geometric Distortion Correction", img)
    key = cv2.waitKey(10)
    if key == ord(' '):     # Space: Pause
        key = cv2.waitKey()
    if key == 27:           # ESC: Exit
        break
    elif key == ord('\t'):  # Tab: Toggle the mode
        show_rectify = not show_rectify

video.release()
cv2.destroyAllWindows()
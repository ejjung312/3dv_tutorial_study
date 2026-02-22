import numpy as np
import cv2

video_file = "../data/chessboard.avi"
K = np.array([[432.7390364738057, 0, 476.0614994349778],
              [0, 431.2395555913084, 288.7602152621297],
              [0, 0, 1]])
dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])
board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

video = cv2.VideoCapture(video_file)
assert video.isOpened(), "Cannot read the given input, " + video_file

box_lower = board_cellsize * np.array([[4, 2, 0], [5, 2, 0], [5, 4, 0], [4, 4, 0]])
box_upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]])

obj_points = board_cellsize * np.array([[c,r,0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

while True:
    valid, img = video.read()
    if not valid:
        break

    success, img_points = cv2.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        # PnP로 pose(외부 파라미터) 추정
        # rvec은 Rodrigues 벡터 3x1
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeff)

        # 3D 점을 2D로 투영
        line_lower, _ = cv2.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv2.projectPoints(box_upper, rvec, tvec, K, dist_coeff)

        cv2.polylines(img, [np.int32(line_lower)], True, (255,0,0), 2)
        cv2.polylines(img, [np.int32(line_upper)], True, (0,0,255), 2)
        for b, t in zip(line_lower, line_upper):
            cv2.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0,255,0), 2)
        
        # Rodrigues 벡터 (3x1) 를 회전 행렬 (3x3)으로 변환
        R, _ = cv2.Rodrigues(rvec)
        # 월드 좌표계로 변환 (C = -R^Tt)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv2.putText(img, info, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    
    cv2.imshow('Pose Estimation (Chessboard)', img)
    key = cv2.waitKey(10)
    if key == ord(' '):
        key = cv2.waitKey()
    if key == 27: # ESC
        break

video.release()
cv2.destroyAllWindows()

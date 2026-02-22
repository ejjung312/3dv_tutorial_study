import numpy as np
import cv2

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name="Camera Calibration"):
    video = cv2.VideoCapture(video_file)
    assert video.isOpened()

    img_select = []
    while True:
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            display = img.copy()
            cv2.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,255,0))
            cv2.imshow(wnd_name, display)

            key = cv2.waitKey(wait_msec)
            if key == ord(' '):
                complete, pts = cv2.findChessboardCorners(img, board_pattern)
                cv2.drawChessboardCorners(display, board_pattern, pts, complete)
                cv2.imshow(wnd_name, display)
                key = cv2.waitKey()
                if key == ord('\r'):
                    img_select.append(img)
            
            if key == 27:
                break
    cv2.destroyAllWindows()
    
    return img_select

def calibrate_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 코너 검출
        complete, pts = cv2.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts) # (N,1,2) 픽셀 좌표
    assert len(img_points) > 0

    # 체커보드의 3D 좌표. 평면이므로 Z=0
    # (0,0,0),(1,0,0),(2,0,0) ...
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    # print("obj_pts=", obj_pts)
    # 좌표를 실제 단위로 변환. 이미지 개수만큼 생성
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`
    # print("obj_points=", obj_points)

    # print("gray.shape=", gray.shape) # (540, 960)
    # print("gray.shape[::-1]=", gray.shape[::-1]) # (960, 540)
    # 카메라 캘리브레이션을 수행해서 내부 파라미터(intrinsic), 왜곡 계수(dist_coeff) 추정
    return cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

if __name__ == "__main__":
    video_file = "../data/chessboard.avi"

    board_pattern = (10, 7) # 체커보드 내부 코너 개수
    board_cellsize = 0.025 # 한 칸 크기 0.025m = 2.5cm

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'

    rms, K, dist_coeff, rvecs, tvecs = calibrate_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')
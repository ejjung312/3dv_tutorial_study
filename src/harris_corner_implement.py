import numpy as np
import matplotlib.pyplot as plt
import cv2

def cornerHarris(img, ksize=3, k=0.04):
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0) # x 방향 gradient
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1) # y 방향 gradient
    # GaussianBlur: ∑w(x,y)⋅Ix2​, 주변 윈도우 내 가중 평균 계산
    M11 = cv2.GaussianBlur(Ix*Ix, (ksize, ksize), 0) # Ix*Ix -> Ix^2
    M22 = cv2.GaussianBlur(Iy*Iy, (ksize, ksize), 0) # Iy*Iy -> Iy^2
    M12 = cv2.GaussianBlur(Ix*Iy, (ksize, ksize), 0) # Ix*Iy -> IxIy
    # M = [[M11, M12], [M12, M22]]
    detM = M11 * M22 - M12 * M12 # 행렬식 계산
    traceM = M11 + M22 # 대각 합
    cornerness = detM - k * traceM**2
    return cornerness

if __name__ == "__main__":
    video = cv2.VideoCapture('../data/chessboard.avi')
    assert video.isOpened()

    while True:
        valid, img = video.read()
        if not valid:
            break
        if img.ndim >= 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        harris = cornerHarris(gray)
        corners = harris > 5e8 # 5x10^8 보다 큰 픽셀만 코너로 판단 (Boolean 배열)

        # dstack: depth 방향 stack -> 채널 방향 결합
        heatmap = np.dstack((np.zeros_like(corners), np.zeros_like(corners), corners*255)) # 코너만 빨간색인 이미지 생성
        # print(heatmap.shape) # (540, 960, 3)
        # 원본 이미지(0.3), 코너 마스크(0.7) 합성
        heatmap = (0.3 * img + 0.7 * heatmap).astype(np.uint8)
        
        cv2.imshow('Harris Cornerness', heatmap)

        key = cv2.waitKey(1)
        if key == ord(' '):
            key = cv2.waitKey()
        if key == 27: # ESC
            break
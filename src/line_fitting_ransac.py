import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def generate_line(pts):
    a = (pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0]) # 기울기 a 계산 (a = (y2-y1)/(x2-x1) )
    b = pts[0][1] - a * pts[0][0] # 절편 계산 (b = y1-ax1)
    """
    y = ax + b
    ax - y + b = 0
    """
    line = np.array([a, -1, b])
    return line / np.linalg.norm(line[:2]) # 점-직선 거리 공식(d) = |ax+by+c| / root(a^2 + b^2)

def evaluate_line(line, p):
    a, b, c = line
    x, y = p
    return np.fabs(a*x + b*y + c) # 실수 전용 절대값 반환 = |ax + by + c|

def fit_line_ransac(data, n_sample, ransac_trial, ransac_threshold):
    best_score = -1
    best_model = None
    for _ in range(ransac_trial):
        sample = random.choices(data, k=n_sample)
        model = generate_line(sample)

        score = 0
        for p in data:
            error = evaluate_line(model, p)
            if error < ransac_threshold:
                score += 1
        if score > best_score:
            best_score = score
            best_model = model

    return best_model, best_score

if __name__ == '__main__':
    true_line = np.array([2, 3, -14]) / np.sqrt(2*2 + 3*3)
    data_range = np.array([-4, 12])
    data_num = 100
    noise_std = 0.2
    outlier_ratio = 0.7

    # line = [a, b, c]
    # y = (ax + c) / -b
    line2y = lambda line, x: (line[0] * x + line[2]) / -line[1]
    # x에 data_range 값을 넣어 y 값 계산 후 작은 순으로 정렬
    y_range = sorted(line2y(true_line, data_range))
    data = []
    for _ in range(data_num):
        x = np.random.uniform(*data_range) # 균등분포에서 랜덤 생성
        if np.random.rand() < outlier_ratio: # 아웃라이어 생성
            y = np.random.uniform(*y_range)
        else:
            y = line2y(true_line, x) # 직선 위 y 값
            x += np.random.normal(scale=noise_std) # 노이즈 추가해서 직선 위 점을 퍼뜨림
            y += np.random.normal(scale=noise_std)
        data.append((x, y))
    data = np.array(data)

    best_line, best_score = fit_line_ransac(data, 2, 100, 0.3)

    # 점에서 직선까지의 L2 거리 제곱 합 최소화
    nnxy = cv2.fitLine(data, cv2.DIST_L2, 0, 0.01, 0.01).flatten() # nnxy = [vx, vy, x0, y0], (vx, vy): 직선의 방향 벡터, (x0, y0): 직선 위의 한 점
    # ax+by+c = 0 -> vy, -vx, -vy*x0 + vx*y0
    # a = vy, b = -vx, c = -vy*x0 + vx*y0
    lsqr_line = np.array([nnxy[1], -nnxy[0], -nnxy[1]*nnxy[2] + nnxy[0] * nnxy[3]])

    plt.plot(data_range, line2y(true_line, data_range), 'r-', label='The true line')
    plt.plot(data[:,0], data[:,1], 'b.', label='Noisy data')
    plt.plot(data_range, line2y(best_line, data_range), 'g-', label=f'RASAC (score={best_score})')
    plt.plot(data_range, line2y(lsqr_line, data_range), 'm-', label='Least square method')
    plt.legend()
    plt.xlim(data_range)
    plt.show()

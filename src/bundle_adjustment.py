import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation  as R
import time
import matplotlib.pyplot as plt
import argparse

f, cx, cy = 1000, 320, 240

msg = """This script is a Python file related to global bundle adjustment. 
Both the least_square and minimum versions of scipy can be used, and the 
least square can also be used with or without the Jacobian matrix. 
By the way, minimizing takes a long time, so please don't think it's impossible and wait.
"""

# Bundle Adjustment에서 Jacobian의 sparsity pattern(희소 구조)
def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3 # 6 DOF * n_cameras + 3 DOF * n_points
    A = lil_matrix((m, n), dtype=int) # m * n 크기의 희소 행렬 생성
    
    # | cam0(6) | cam1(6) | ... | pt0(3) | pt1(3) | pt2(3) | ...

    i = np.arange(camera_indices.size)
    for s in range(6): # 카메라 블록 (6 DOF)
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    
    for s in range(3): # 3D 포인트 블록 (3 DOF)
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def project(points, cam_trans):
    global f, cx, cy
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    
    rot_vecs = cam_trans[:, :3] #
    t_vecs = cam_trans[:, 3:]
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    points_proj = cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    points_proj += t_vecs
    points_proj = points_proj @ K.T 
    points_proj /= points_proj[:, 2, np.newaxis] # 
    
    return points_proj[:, :2].ravel()

def func2(params, points_2d, n_cameras, n_points, camera_indices, point_indices):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))        
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    result = (points_2d - points_proj).ravel()
    return result

def func3(params, points_2d, n_cameras, n_points, camera_indices, point_indices):
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))   
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    result = np.sum(((points_2d - points_proj)**2).ravel())
    return result

def main():
    global f, cx, cy
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument('--jac', required=False, default=True, help='Using Jacobian to boost up optimizer')
    parser.add_argument('--method', required=False, default='least_square' , help='Choose between least_square and minimize. Only two things are suppported')
    parser.add_argument('--show-time', required=False, default=False, help='Show how much time is spended for calculating')
    parser.add_argument('--show-test', required=False, default=False, help='Show test points for check real points')

    args = parser.parse_args()
    METHOD = args.method
    JAC = args.jac
    SHOW_TIME = args.show_time
    SHOW_TEST = args.show_test

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    input_num = 5

    xs = []
    for i in range(input_num):
        input = f"../data/image_formation{i}.xyz"
        x = np.genfromtxt(input, delimiter=" ")
        x = x[:, :2]
        xs.append(x)
    
    xs = np.array(xs)

    n_cameras = xs.shape[0] # 5
    n_points = xs.shape[1] # 160
    
    cam_indices = np.array([])
    length_c_ind = np.arange(n_points, dtype=int) # 0~159
    # print(length_c_ind.shape) (160, )
    
    for i in range(n_cameras):
        # np.full_like(length_c_ind, i): length_c_ind 와 같은 shape을 가지는 배열에 모든 값을 i로 채움
        # np.hstack: 1차원 배열이면 cam_indices 뒤에 이어 붙임
        cam_indices = np.hstack((cam_indices, np.full_like(length_c_ind, i)))
    
    point_indices = np.array([])
    for i in range(n_cameras):
        # np.linspace(0, 159, 160, dtype=int): 0부터 159까지 160개 간격으로 배열 생성
        point_indices = np.hstack((point_indices, np.linspace(0, 159, 160, dtype=int))) # (0~159) * 5
    # print(point_indices)

    cam_indices = cam_indices.astype(int)
    point_indices = point_indices.astype(int)

    cameras = np.zeros((xs.shape[0], 6)) # 0을로 채워진 (5,6) -> 5개의 R, t
    cameras[:, 2] = 1 # z축에 1 radian을 설정해 초기값 설정 - BA에서 많이 쓰임
    # (xs.shape[1], xs.shape[2]+1) = (160, 3) -> 3D 좌표를 만듬
    # [0,0,5.5]로 채운 3D 좌표 (모든 3D 포인트를 카메라 앞 5.5m 지점에 초기 배치)
    Xs = np.full((xs.shape[1], xs.shape[2]+1), np.array([[0,0,5.5]]))
    x0 = np.hstack((cameras.ravel(), Xs.ravel())) # 비선형 최적화의 초기 파라미터 벡터
    # print(xs.shape) # (5, 160, 2)
    xs = xs.ravel()
    # print(xs.shape) # (1600, )

    if METHOD == 'least_square':
        if JAC:
            J = bundle_adjustment_sparsity(n_cameras=n_cameras, n_points=n_points, camera_indices=cam_indices, point_indices=point_indices)
        else:
            J = None
        
        if SHOW_TIME:
            t = time.time()
            res = least_squares(func2, x0, verbose=2, ftol=1e-15, method='trf', jac_sparsity=J, args=(xs, n_cameras, n_points, cam_indices, point_indices))
            print("total time:",time.time() - t)
        else:
            res = least_squares(func2, x0, verbose=2, ftol=1e-15, method='trf', jac_sparsity=J, args=(xs, n_cameras, n_points, cam_indices, point_indices))

    elif METHOD == 'minimize':
        res = minimize(func3, x0, args=(xs, n_cameras, n_points, cam_indices, point_indices))
    
    opt_cameras = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    opt_points = res.x[n_cameras*6:].reshape((n_points, 3))

    if SHOW_TEST:
        for i in range(len(opt_cameras)):
            test_rvec = R.from_rotvec(opt_cameras[i, :3]) # theta x, y ,z translation x, y, z
            test_tvec = opt_cameras[i, 3:]
            test_points = opt_points[0] # 첫번째 x, y, z로 테스트하기.
            a = test_rvec.apply(test_points) + test_tvec
            a = a @ K.T
            a /= a[2]
            print(a)

    f0 = func2(x0, xs, n_cameras, n_points, cam_indices, point_indices)
    plt.plot(f0, 'r', label='with_err')
    plt.plot(res.fun, 'b', label='no_err')
    plt.legend()
    plt.show()
    # Store the 3D points to an XYZ file
    point_file = "bundle_adjustment_global(point)_by_cjh.xyz"
    with open(point_file, 'wt') as f:
        for i in range(n_points):
            data = f"{opt_points[i, 0]} {opt_points[i, 1]} {opt_points[i, 2]}\n"
            f.write(data)

    camera_file = "bundle_adjustment_global(camera)_by_cjh.xyz"
    with open(camera_file, 'wt') as f:
        for i in range(n_cameras):
            data = f"{opt_cameras[i, 0]} {opt_cameras[i, 1]} {opt_cameras[i, 2]} {opt_cameras[i, 3]} {opt_cameras[i, 4]} {opt_cameras[i, 5]}\n"
            f.write(data)

if __name__ == "__main__":
    main()
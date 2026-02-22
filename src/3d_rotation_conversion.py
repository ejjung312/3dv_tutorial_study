import numpy as np
from scipy.spatial.transform import Rotation

# 3d rotation
euler = (45, 30, 60) # XYZ 순서. 도 단위

# rotation object 생성
robj = Rotation.from_euler('zyx', euler[::-1], degrees=True) # 역순으로 뒤집음 euler[start:stop:step]

print('\n## Euler Angle (ZYX)')
print(np.rad2deg(robj.as_euler('zyx'))) # [60. 30. 45.]

print('\n## Rotation Matrix')
print(robj.as_matrix())

print('\n## Rotation Vector')
print(robj.as_rotvec())

print('\n## Quaternion (XYZW)')
print(robj.as_quat())
import cv2
import math
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import apriltag


options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)



pose_x = []
pose_y = []
pose_z = []
points_3D = [[-500, 0, 0], [0, 500, 0], [500, 0, 0], [0, -500, 0]]
points_3D = np.array(points_3D, dtype="double")
camera_matrix = [[1.661923585022047e+03, 0, 9.566273843667578e+02], [0, 1.661966349223287e+03, 5.328345956293313e+02], [0, 0, 1]]
camera_matrix = np.array(camera_matrix, dtype="double")
dist = np.zeros((4, 1), dtype="double")
P_old = []
A, B, C, D = 0, 0, 0, 0
counter = 0

def get_euler_angle(rotation_vector):
    # 旋转顺序是z,y,x，对于相机来说就是滚转，偏航，俯仰
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    return roll, pitch, yaw

def rotate_by_z(x, y, theta_z):
    outx = math.cos(theta_z) * x - math.sin(theta_z) * y
    outy = math.sin(theta_z) * x + math.cos(theta_z) * y
    return outx, outy


def rotate_by_x(y, z, theta_x):
    outy = math.cos(theta_x) * y - math.sin(theta_x) * z
    outz = math.sin(theta_x) * y + math.cos(theta_x) * z
    return outy, outz


def rotate_by_y(z, x, theta_y):
    outz = math.cos(theta_y) * z - math.sin(theta_y) * x
    outx = math.sin(theta_y) * z + math.cos(theta_y) * x
    return outz, outx


vc = cv2.VideoCapture(0)
first_frame = True
if vc.isOpened():
    op = True
else:
    op = False

while op:
    ret, frame = vc.read()
    if frame is None:
        break

    # 将帧转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测Apriltag
    results = detector.detect(gray)
    P_ = []
    for r in results:
        # 获取标记的角点
        pts = r.corners
        # 将角点坐标转换为整数
        pts = pts.astype(int)
        # 绘制四个角点
        for i in range(4):
            cv2.circle(frame, (pts[i][0], pts[i][1]), 3, (0, 0, 255), -1)
            P_ = np.array(pts[i][0],pts[i][1])


        cv2.circle(frame, (pts[0][0], pts[0][1]), 3, (0, 0, 255), -1)
        cv2.circle(frame, (pts[1][0], pts[1][1]), 3, (0, 0, 255), -1)
        cv2.circle(frame, (pts[2][0], pts[2][1]), 3, (0, 0, 255), -1)
        cv2.circle(frame, (pts[3][0], pts[3][1]), 3, (0, 0, 255), -1)

        A = np.array([pts[0][0],pts[0][1]])
        B = np.array([pts[1][0],pts[1][1]])
        C = np.array([pts[2][0],pts[2][1]])
        D = np.array([pts[3][0],pts[3][1]])
        
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(frame_hsv)

    try:
        P_ = [A, B, C, D]
        points_2D = np.array(P_, dtype="double")
        ret, rvec, tvec = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        (Z_end, jacobian_z) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rvec, tvec, camera_matrix, dist)
        (X_end, jacobian_x) = cv2.projectPoints(np.array([(500.0, 0.0, 0.0)]), rvec, tvec, camera_matrix, dist)
        (Y_end, jacobian_y) = cv2.projectPoints(np.array([(0.0, 500.0, 0.0)]), rvec, tvec, camera_matrix, dist)
        (O, jacobian_o) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec, tvec, camera_matrix, dist)
        Z_end = (int(Z_end[0][0][0]), int(Z_end[0][0][1]))
        X_end = (int(X_end[0][0][0]), int(X_end[0][0][1]))
        Y_end = (int(Y_end[0][0][0]), int(Y_end[0][0][1]))
        O_ = (int(O[0][0][0]), int(O[0][0][1]))
        cv2.line(frame, O_, X_end, (50, 255, 50), 10)
        cv2.line(frame, O_, Y_end, (50, 50, 255), 10)
        cv2.line(frame, O_, Z_end, (255, 50, 50), 10)
        cv2.putText(frame, "X", X_end, cv2.FONT_HERSHEY_SIMPLEX, 3, (5, 150, 5), 10)
        cv2.putText(frame, "Y", Y_end, cv2.FONT_HERSHEY_SIMPLEX, 3, (5, 5, 150), 10)
        cv2.putText(frame, "Z", Z_end, cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 5, 5), 10)

        cv2.circle(frame, A, 3, (255, 255, 255), 10)
        cv2.circle(frame, B, 3, (255, 255, 255), 10)
        cv2.circle(frame, C, 3, (255, 255, 255), 10)
        cv2.circle(frame, D, 3, (255, 255, 255), 10)

        #cv2.putText(frame, "A", A + [15, 15], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        #cv2.putText(frame, "B", B + [15, 15], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        #cv2.putText(frame, "C", C + [15, 15], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        #cv2.putText(frame, "D", D + [15, 15], cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

        xt, yt, zt = tvec
        xr, yr, zr = rvec
        rvec_str = 'rotation_vector: ({0}, {1}, {2})'
        rvec_str = rvec_str.format(float(xr), float(yr), float(zr))
        tvec_str = 'translation_vector: ({0}, {1}, {2})'
        tvec_str = tvec_str.format(float(xt), float(yt), float(zt))

        #cv2.putText(frame, tvec_str, [30, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(frame, rvec_str, [30, 80], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        roll, pitch, yaw = get_euler_angle(rvec)
        # print(roll, pitch, yaw)

        Xc, Yc, Zc = tvec
        Xc, Yc = rotate_by_z(Xc, Yc, -roll)
        Zc, Xc = rotate_by_y(Zc, Xc, -yaw)
        Yc, Zc = rotate_by_x(Yc, Zc, -pitch)


        cv2.putText(frame, 'x : {}'.format(-Xc), [0, 30], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, 'y : {}'.format(-Yc), [0, 80], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, 'y : {}'.format(-Zc), [0, 130], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        counter = counter + 1
    except:
        print("no points")
    cv2.imshow("res", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break




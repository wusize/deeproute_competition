import json
import os
from types import coroutine 
import numpy as np
import mayavi.mlab as mlab
from numpy.core.defchararray import center
from numpy.lib.function_base import append
import pandas as pd
import math

class_names=[
            'CAR',
            'VAN',
            'TRUCK',
            'BIG_TRUCK', 
            'BUS',
            'PEDESTRIAN',
            'CYCLIST',
            'TRICYCLE',
            'CONE']

class_to_color = {'CAR': (1,0,0), 'VAN':(1,1,0), 'TRUCK':(0,1,0), 'BIG_TRUCK':(0,1,1), 'BUS':(1,0,1),
'PEDESTRIAN':(0,0,1), 'CYCLIST':(0,0.5,0.5), 'TRICYCLE':(0.5, 0, 0.5), 'CONE':(0.5, 0.5, 0.5)}

def center_to_box3d_corners(c_y, c_x, c_z, w, l, h, head):
    xs = []
    ys = []
    zs = []

    x_0 = c_x - 0.5 * l * math.cos(head) + 0.5 * w * math.sin(head)
    y_0 = c_y - 0.5 * l * math.sin(head) - 0.5 * w * math.cos(head)
    z_0 = c_z - h * 0.5
    xs.append(x_0)
    ys.append(y_0)
    zs.append(z_0)

    x_1 = c_x + 0.5 * l * math.cos(head) + 0.5 * w * math.sin(head)
    y_1 = c_y + 0.5 * l * math.sin(head) - 0.5 * w * math.cos(head)
    z_1 = c_z - h * 0.5
    xs.append(x_1)
    ys.append(y_1)
    zs.append(z_1)

    x_2 = c_x + 0.5 * l * math.cos(head) - 0.5 * w * math.sin(head)
    y_2 = c_y + 0.5 * l * math.sin(head) + 0.5 * w * math.cos(head)
    z_2 = c_z - h * 0.5
    xs.append(x_2)
    ys.append(y_2)
    zs.append(z_2)

    x_3 = c_x - 0.5 * l * math.cos(head) - 0.5 * w * math.sin(head)
    y_3 = c_y - 0.5 * l * math.sin(head) + 0.5 * w * math.cos(head)
    z_3 = c_z - h * 0.5
    xs.append(x_3)
    ys.append(y_3)
    zs.append(z_3)

    x_4 = c_x - 0.5 * l * math.cos(head) + 0.5 * w * math.sin(head)
    y_4 = c_y - 0.5 * l * math.sin(head) - 0.5 * w * math.cos(head)
    z_4 = c_z + h * 0.5
    xs.append(x_4)
    ys.append(y_4)
    zs.append(z_4)

    x_5 = c_x + 0.5 * l * math.cos(head) + 0.5 * w * math.sin(head)
    y_5 = c_y + 0.5 * l * math.sin(head) - 0.5 * w * math.cos(head)
    z_5 = c_z + h * 0.5
    xs.append(x_5)
    ys.append(y_5)
    zs.append(z_5)

    x_6 = c_x + 0.5 * l * math.cos(head) - 0.5 * w * math.sin(head)
    y_6 = c_y + 0.5 * l * math.sin(head) + 0.5 * w * math.cos(head)
    z_6 = c_z + h * 0.5
    xs.append(x_6)
    ys.append(y_6)
    zs.append(z_6)

    x_7 = c_x - 0.5 * l * math.cos(head) - 0.5 * w * math.sin(head)
    y_7 = c_y - 0.5 * l * math.sin(head) + 0.5 * w * math.cos(head)
    z_7 = c_z + h * 0.5
    xs.append(x_7)
    ys.append(y_7)
    zs.append(z_7)

    corner = [xs, ys, zs]
    return corner

def plot3Dbox(corners, class_name):
    for i in range(corners.shape[0]):
        corner = corners[i]
        idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
        x = corner[0, idx]
        y = corner[1, idx]
        z = corner[2, idx]
        # mayavi.mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', representation='wireframe', line_width=5)
        mlab.plot3d(x, y, z, color=class_to_color[class_name], colormap='Spectral', representation='wireframe', line_width=1)
    #mlab.show()

label_path = 'D:/from_wusize/gt/00001.txt'
lidar_path = label_path.replace('gt', 'pc').replace('.txt', '.bin')
data = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 4])
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
pointcloud = data_pd.values  # 初始化储存数据的array

x = pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point

r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

vals = 'height'  #按照什么变颜色
if vals == "height":
    col = z
else:
    col = d

fig = mlab.figure(bgcolor=(1, 1, 1), size=(640, 500))   # 111没点的地方是白色 000没点的地方是黑色
mlab.points3d(x, y, z,
                     col,  # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot', 'spectral'
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )

info = json.loads(open(label_path, 'r').read())

CAR_corners = []
VAN_corners = []
TRUCK_corners = []
BIG_TRUCK_corners = []
BUS_corners = []
PEDESTRIAN_corners = []
CYCLIST_corners = []
TRICYCLE_corners = []
CONE_corners = []
for i in range(len(info['objects'])):

    c_y = info['objects'][i]['position']['y']
    c_x = info['objects'][i]['position']['x']
    c_z = info['objects'][i]['position']['z']

    w = info['objects'][i]['bounding_box']['width']
    l = info['objects'][i]['bounding_box']['length']
    h = info['objects'][i]['bounding_box']['height']

    heading = info['objects'][i]['heading']

    s_x = math.cos(heading)
    s_y = math.sin(heading)
    mlab.quiver3d(c_x, c_y, c_z, s_x, s_y, 0, color=(0, 0, 0), line_width=1.5,scale_factor=1)


    corner = center_to_box3d_corners(c_y, c_x, c_z, w, l, h, heading)

    cls = info['objects'][i]['type']
    score = info['objects'][i]['score']
    p = cls + '_' + str(score)

    mlab.text3d(corner[0][4], corner[1][4], corner[2][4], p, color=(0,0,0), line_width=0.04, scale=0.25)

    if cls == class_names[0]:
        CAR_corners.append(corner)
    elif cls == class_names[1]:
        VAN_corners.append(corner)
    elif cls == class_names[2]:
        TRUCK_corners.append(corner)
    elif cls == class_names[3]:
        BIG_TRUCK_corners.append(corner)
    elif cls == class_names[4]:
        BUS_corners.append(corner)
    elif cls == class_names[5]:
        CYCLIST_corners.append(corner)
    elif cls == class_names[6]:
        TRICYCLE_corners.append(corner)
    else:
        CONE_corners.append(corner)

CAR_corners = np.array(CAR_corners)
VAN_corners = np.array(VAN_corners)
TRUCK_corners = np.array(TRUCK_corners)
BIG_TRUCK_corners = np.array(BIG_TRUCK_corners)
BUS_corners = np.array(BUS_corners)
PEDESTRIAN_corners = np.array(PEDESTRIAN_corners)
CYCLIST_corners = np.array(CYCLIST_corners)
TRICYCLE_corners = np.array(TRICYCLE_corners)
CONE_corners = np.array(CONE_corners)

plot3Dbox(CAR_corners, 'CAR')
plot3Dbox(VAN_corners, 'VAN')
plot3Dbox(TRUCK_corners, 'TRUCK')
plot3Dbox(BIG_TRUCK_corners, 'BIG_TRUCK')
plot3Dbox(BUS_corners, 'BUS')
plot3Dbox(PEDESTRIAN_corners, 'PEDESTRIAN')
plot3Dbox(CYCLIST_corners, 'CYCLIST')
plot3Dbox(TRICYCLE_corners, 'TRICYCLE')
plot3Dbox(CONE_corners, 'CONE')
mlab.show()


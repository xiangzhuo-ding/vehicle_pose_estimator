import cv2
import numpy as np
from math import sin, cos
from scipy.optimize import minimize
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import pandas as pd
from sklearn.linear_model import LinearRegression

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

PATH = '../data/'
train = pd.read_csv(PATH + 'train.csv')
points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train['PredictionString']:
        coords = str2coords(ps)
        arr += [c[col] for c in coords]
    points_df[col] = arr


xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img



def get_img_coords(s):
    # From camera.zip
    camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                            [0, 2305.8757, 1354.9849],
                            [0, 0, 1]], dtype=np.float32)
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    return image

def visualize(img, coords):
    # From camera.zip
    camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                            [0, 2305.8757, 1354.9849],
                            [0, 0, 1]], dtype=np.float32)
    camera_matrix_inv = np.linalg.inv(camera_matrix)
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img


def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x

def get_peak(heatmap, threshold):
    peak = np.zeros((heatmap.shape[0], heatmap.shape[1]))
    for i in range(1,heatmap.shape[0]-1):
        for j in range(1, heatmap.shape[1]-1):
            if heatmap[i][j] > threshold:
                
                if heatmap[i][j] < heatmap[i-1][j-1]:
                    continue

                if heatmap[i][j] < heatmap[i][j-1]:
                    continue

                if heatmap[i][j] < heatmap[i+1][j-1]:
                    continue

                if heatmap[i][j] < heatmap[i-1][j]:
                    continue

                if heatmap[i][j] < heatmap[i+1][j]:
                    continue

                if heatmap[i][j] < heatmap[i-1][j+1]:
                    continue

                if heatmap[i][j] < heatmap[i][j+1]:
                    continue

                if heatmap[i][j] < heatmap[i+1][j+1]:
                    continue

                peak[i][j] = 1
    return peak

def extract_coords(prediction, threshold=0.1):
    logits = prediction[0]
    # logits_prob = torch.sigmoid(torch.tensor(logits)).numpy()
    regr_output = prediction[1:]
    points = np.argwhere(get_peak(logits, threshold) == 1)
    
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = optimize_xy(r, c, coords[-1]['x'], coords[-1]['y'], coords[-1]['z'])
    # coords = clear_duplicates(coords)
    return coords


def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict


def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy


def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        IMG_WIDTH = 1024
        IMG_HEIGHT = IMG_WIDTH // 16 * 5
        MODEL_SCALE = 8
        IMG_SHAPE = (2710, 3384, 3)
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x-r)**2 + (y-c)**2)#  + max(0.4, slope_err)
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new


# def optimize_xy(r, c, x0, y0, z0):
#     def distance_fn(xyz):
#         IMG_WIDTH = 1024
#         IMG_HEIGHT = IMG_WIDTH // 16 * 5
#         MODEL_SCALE = 8
#         IMG_SHAPE = (2710, 3384, 3)

#         x, y, z = xyz
#         x, y = convert_3d_to_2d(x, y, z)
#         y, x = x, y
#         x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
#         y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] *4/3) / MODEL_SCALE
#         return max(0.2, (x-r)**2 + (y-c)**2)
    
#     res = minimize(distance_fn, [x0, y0, z0], method='Powell')
#     x_new, y_new, z_new = res.x
#     return x_new, y_new, z_new

def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < 5:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]



def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')

    return regr_dict



def preprocess_image(img, flip=False):
    IMG_WIDTH = 1024
    IMG_HEIGHT = IMG_WIDTH // 16 * 5

    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def get_mask_and_regr(img, labels, flip=False):
    IMG_WIDTH = 1024
    IMG_HEIGHT = IMG_WIDTH // 16 * 5
    MODEL_SCALE = 8

    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4/3) / MODEL_SCALE
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
    return mask, regr

class canvas(object):
    def __init__(self, close = True):
        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(111)
        self.t = 0
        self.close = close

    def bird(self, points_df, show = True):
        self.t = not self.t
        if self.close:
            plt.cla()
            plt.ion()
        road_width = 20
        road_xs = [-road_width, road_width, road_width, -road_width, -road_width]
        road_ys = [0, 0, 500, 500, 0]


        plt.axes().set_aspect(1)
        plt.xlim(-50,50)
        plt.ylim(0,60)

        # View road
        plt.fill(road_xs, road_ys, alpha=0.2, color='gray')
        plt.plot([road_width/2,road_width/2], [-10*self.t,150], alpha=0.4, linewidth=4, color='white', ls='--')
        plt.plot([-road_width/2,-road_width/2], [-10*self.t,150], alpha=0.4, linewidth=4, color='white', ls='--')
        # View cars
        x = np.array([point['x'] for point in points_df])
        y = np.array([point['y'] for point in points_df])
        z = np.array([point['z'] for point in points_df])
        Y = np.sqrt(z**2 + y**2)
        pitch = np.array([point['pitch'] for point in points_df])

        for i in range(len(x)):
            t = mpl.transforms.Affine2D().rotate_deg_around(x[i],Y[i],pitch[i]*180/np.pi) + self.ax.transData
            rect = patches.Rectangle((x[i]-1,Y[i]-2), 2, 4, color="blue", alpha=0.50, transform = t)
            self.ax.add_patch(rect)
            # arrow = patches.Arrow(x[i], Y[i], 3*np.cos(pitch[i]+np.pi/2), 3*np.sin(pitch[i]+np.pi/2), 1.5,color = "red")
            # self.ax.add_patch(arrow)

        plt.plot([0,48], [0,60], [0,-48], [0,60], color = 'red', ls = '--')


        plt.scatter(x, Y, color='red', s=10)
        if show:
            plt.show()
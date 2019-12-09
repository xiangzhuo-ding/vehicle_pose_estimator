import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc
import os

import argparse 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datasets.baseline import baseline
from models.BaselineModel import MyUNet, LargeUNet
from models.AttentionModel import AttentionUnet, EAUNet
from utils.preprocess import *
import torch
from torch import nn


PATH = '../data/'
os.listdir(PATH)

train_images_dir = PATH + 'train_images/{}.jpg'
# test_images_dir = PATH + 'test_images/{}.jpg'

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')

df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
df_test = test
dev_dataset = baseline(df_dev, train_images_dir, training=False)
train_dataset = baseline(df_train, train_images_dir, training=False)

parser = argparse.ArgumentParser(description='PKU')



args = parser.parse_args()
args.cuda = True
print(args)

model = EAUNet(8, args).cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('./saved_models/15.pth'))
model.eval()




torch.cuda.empty_cache()
gc.collect()

for idx in range(20):
    img, mask, regr, meta = dev_dataset[idx]
    output = model(torch.tensor(img[None]).cuda()).data.cpu().numpy()

    coords_pred = extract_coords(output[0], threshold = 0.001)
    coords_true = extract_coords(np.concatenate([mask[None], regr], 0), threshold = 0.0)
    
    #print(coords_pred)
    #print(coords_true)

    img = imread(train_images_dir.format(df_dev['ImageId'].iloc[idx]))

    fig, axes = plt.subplots(1, 2, figsize=(30,30))
    axes[0].set_title('Ground truth')
    axes[0].imshow(visualize(img, coords_true))
    axes[1].set_title('Prediction')
    axes[1].imshow(visualize(img, coords_pred))
    # plt.show()

    

    plot1 = canvas(False)
    plot1.bird(coords_pred, False)

    plot2 = canvas(False)
    plot2.bird(coords_true)



import pandas as pd
import matplotlib.pyplot as plt
import gc
import os

import argparse 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datasets.baseline import baseline
from models.BaselineModel import MyUNet
from utils.preprocess import *
import torch

# from Model import *
# from Utils import *

PATH = '../data/'
os.listdir(PATH)

train_images_dir = PATH + 'train_images/{}.jpg'
# test_images_dir = PATH + 'test_images/{}.jpg'

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')

df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
df_test = test
dev_dataset = baseline(df_dev, train_images_dir, training=False)


parser = argparse.ArgumentParser(description='PKU')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model-path', type=str, default='./save_models/{}.pth',
                    help='save train models')
parser.add_argument('--loss-path', type=str, default='./save_models/loss.csv',
                    help='save losses')
parser.add_argument('--load-model', action='store_true', default=False,
                    help='load model') 
parser.add_argument('--model-name', type=str, default='',
                    help='load model name') 
parser.add_argument('--start-epoch', type=int, default=0, metavar='SP',
                    help='starting epoch (default: 0)') 
parser.add_argument('--batch-size', type=float, default=4, metavar='SP',
                    help='batch size (default: 4)')  


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
model = MyUNet(8, args)
model.load_state_dict(torch.load('./saved_models/9.pth'))
model.cuda().eval()




torch.cuda.empty_cache()
gc.collect()

for idx in range(8):
    img, mask, regr = dev_dataset[idx]
    
    output = model(torch.tensor(img[None]).cuda()).data.cpu().numpy()
    # print(output.shape)
    coords_pred = extract_coords(output[0])
    coords_true = extract_coords(np.concatenate([mask[None], regr], 0))
    
    img = imread(train_images_dir.format(df_dev['ImageId'].iloc[idx]))
    print(img.shape)

    fig, axes = plt.subplots(1, 2, figsize=(30,30))
    axes[0].set_title('Ground truth')
    axes[0].imshow(visualize(img, coords_true))
    axes[1].set_title('Prediction')
    axes[1].imshow(visualize(img, coords_pred))
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import gc
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Model import *
from Utils import *

PATH = '../data/'
os.listdir(PATH)

train_images_dir = PATH + 'train_images/{}.jpg'
# test_images_dir = PATH + 'test_images/{}.jpg'

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')

df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
df_test = test
dev_dataset = CarDataset(df_dev, train_images_dir, training=False)


model = torch.load('./Model/test.pth')
model.eval()

# train = pd.read_csv('../data/train.csv')
# img0 = imread('../data/train_images/' + train['ImageId'][0] + '.jpg')
# img = preprocess_image(img0)
# img = np.rollaxis(img,2,0)

# output = model(torch.tensor(img[None]).cuda())
# logits = output[0,0].data.cpu().numpy()

# plt.figure(figsize=(16,16))
# plt.title('Model predictions')
# plt.imshow(logits)
# plt.show()

# plt.figure(figsize=(16,16))
# plt.title('Model predictions thresholded')
# plt.imshow(logits > 0)
# plt.show()


torch.cuda.empty_cache()
gc.collect()

for idx in range(8):
    img, mask, regr = dev_dataset[idx]
    
    output = model(torch.tensor(img[None]).cuda()).data.cpu().numpy()
    # print(output.shape)
    coords_pred = extract_coords(output[0])
    coords_true = extract_coords(np.concatenate([mask[None], regr], 0))
    
    img = imread(train_images_dir.format(df_dev['ImageId'].iloc[idx]))
    
    fig, axes = plt.subplots(1, 2, figsize=(30,30))
    axes[0].set_title('Ground truth')
    axes[0].imshow(visualize(img, coords_true))
    axes[1].set_title('Prediction')
    axes[1].imshow(visualize(img, coords_pred))
    plt.show()

import pandas as pd
from datasets.baseline import baseline
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class load_data:
    def __init__(self, path = '../data/', 
                        train = 'train.csv', 
                        test = 'sample_submission.csv'):
        self.path = path
        self.train_img_dir = path + 'train_images/{}.jpg'
        self.test_img_dir = path + 'test_images/{}.jpg'

        self.train = pd.read_csv(path + train)
        self.test = pd.read_csv(path + test) 

    def get_baseline(self, rs = 42, test_size = 0.01, b_size = 2):
        df_train, df_dev = train_test_split(self.train, test_size=test_size, random_state=rs)

        train_dataset = baseline(df_train, self.train_img_dir, training=True)
        dev_dataset = baseline(df_dev, self.train_img_dir, training=False)
        test_dataset = baseline(self.test, self.test_img_dir, training=False)
    

        train_loader = DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=True, num_workers=4)
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=b_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=b_size, shuffle=False, num_workers=1)

        return train_loader, dev_loader, test_loader


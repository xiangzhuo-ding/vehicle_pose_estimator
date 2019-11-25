import cv2
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


import gc


import argparse 
from models.BaselineModel import MyUNet
import torch
from utils.preprocess import *





def main():
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

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

    cv2.namedWindow("Preview")

    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        
    while rval:
        frame = cv2.resize(frame,(3384,2710))

        # frame =  cv2.imread('../data/train_images/ID_0a1eb2c76.jpg')
        # print(frame.shape)

        img = preprocess_image(frame.copy())
        img = np.rollaxis(img,2,0)

        output = model(torch.tensor(img[None]).cuda()).data.cpu().numpy()
        coords_pred = extract_coords(output[0])
        print(coords_pred)
        

        img = visualize(frame, coords_pred)
        img = cv2.resize(img,(640,480))


        cv2.imshow("Preview",img)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
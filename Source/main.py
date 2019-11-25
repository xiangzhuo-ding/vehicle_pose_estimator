from datasets.pku import load_data
import argparse 
from models.BaselineModel import MyUNet
import torch
from utils.trainer import *
from torch import optim
from torch.optim import lr_scheduler
import pandas as pd

def TrainModel(args):
    print("Training model")
    data = load_data()
    train_loader, dev_loader, test_loader = data.get_baseline(b_size=args.batch_size)

    model = MyUNet(args.output_size, args)

    if args.load_model:
        print("Loading model: " +  args.model_name)
        model_path = args.model_path.format(args.model_name)
        model.load_state_dict(torch.load(model_path))

    if args.cuda:
        print('\nGPU is ON!')
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, 
                                        step_size=max(args.epochs, 10) * len(train_loader) // 3, 
                                        gamma=0.1)

    train(model, optimizer, exp_lr_scheduler, train_loader, dev_loader, args)


def Inference(args):
    print("Running Inference")
    data = load_data()
    train_loader, dev_loader, test_loader = data.get_baseline(b_size=args.batch_size)

    if args.load_model:
        print("Loading model: " +  args.model_name)
        model_path = args.model_path.format(args.model_name)
        model = MyUNet(8, args)
        model.load_state_dict(torch.load(model_path))
    
    if args.cuda:
        print('\nGPU is ON!')
        model = model.cuda()
    
    history = pd.DataFrame()
    evaluate_model(model, args.start_epoch, dev_loader, history, args.cuda)
    history.to_csv('./validation/' + args.model_name + "_loss.csv")

    

# Training settings
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
parser.add_argument('--batch-size', type=int, default=4, metavar='BZ',
                    help='batch size (default: 4)')   
parser.add_argument('--inference', action='store_true', default=False,
                    help='run analysis for the validation set)') 
parser.add_argument('--output-size', type=float, default=8, metavar='OS',
                    help='output-size (default: 8)')                  

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if __name__ == "__main__":
    if not args.inference:
        TrainModel(args)
    else:
        Inference(args)

from tqdm import tqdm
from models.Loss import FocalLoss, EvaluationLoss
import torch
import pandas as pd

def train(model, optimizer, exp_lr_scheduler, train_loader, dev_loader, args):
    history = pd.DataFrame()
    for epoch in range(args.start_epoch, args.epochs):
        train_model(model, optimizer, exp_lr_scheduler, epoch, train_loader, history, args.cuda)
        torch.save(model.state_dict(), args.model_path.format(epoch))
        evaluate_model(model, epoch, dev_loader, history, args.cuda)
        history.to_csv(args.loss_path)


def train_model(model, optimizer, exp_lr_scheduler, epoch, train_loader, history=None, cuda=True):
    model.train()
    loss = 0
    
    for batch_idx, (img_batch, mask_batch, regr_batch, meta) in enumerate(tqdm(train_loader)):
        if cuda:
            img_batch = img_batch.cuda()
            mask_batch = mask_batch.cuda()
            regr_batch = regr_batch.cuda()
        
        optimizer.zero_grad()
        output = model(img_batch)

        loss = FocalLoss(output, mask_batch, regr_batch)
        extra_loss = EvaluationLoss(output, meta[0])

        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'distance_loss'] = extra_loss['distance_loss']
            history.loc[epoch + batch_idx / len(train_loader), 'yaw_loss'] = extra_loss['yaw_loss']
            history.loc[epoch + batch_idx / len(train_loader), 'pitch_loss'] = extra_loss['pitch_loss']
            history.loc[epoch + batch_idx / len(train_loader), 'roll_loss'] = extra_loss['roll_loss']
            history.loc[epoch + batch_idx / len(train_loader), 'x_loss'] = extra_loss['x_loss']
            history.loc[epoch + batch_idx / len(train_loader), 'y_loss'] = extra_loss['y_loss']
            history.loc[epoch + batch_idx / len(train_loader), 'z_loss'] = extra_loss['z_loss']

        loss.backward()
        
        optimizer.step()
        exp_lr_scheduler.step()

    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))
    

def evaluate_model(model, epoch, dev_loader, history=None, cuda=True):
    model.eval()
    loss = 0

    total_loss = dict()
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch, meta in tqdm(dev_loader):
            if cuda:
                img_batch = img_batch.cuda()
                mask_batch = mask_batch.cuda()
                regr_batch = regr_batch.cuda()

            output = model(img_batch)

            loss += FocalLoss(output, mask_batch, regr_batch, size_average=False).data
            extra_loss = EvaluationLoss(output, meta[0])

            for x in extra_loss:
                if x not in total_loss:
                    total_loss[x] = extra_loss[x]
                else:
                    total_loss[x] += extra_loss[x]

    loss /= len(dev_loader.dataset)
    for x in total_loss:
        total_loss[x] /= len(dev_loader.dataset)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
        history.loc[epoch, 'train_loss'] = loss.data.cpu().numpy()
        history.loc[epoch, 'distance_loss'] = extra_loss['distance_loss']
        history.loc[epoch, 'yaw_loss'] = extra_loss['yaw_loss']
        history.loc[epoch, 'pitch_loss'] = extra_loss['pitch_loss']
        history.loc[epoch, 'roll_loss'] = extra_loss['roll_loss']
        history.loc[epoch, 'x_loss'] = extra_loss['x_loss']
        history.loc[epoch, 'y_loss'] = extra_loss['y_loss']
        history.loc[epoch, 'z_loss'] = extra_loss['z_loss']
    print('Dev loss: {:.4f}'.format(loss))

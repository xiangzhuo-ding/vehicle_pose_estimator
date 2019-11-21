from tqdm import tqdm
from models.Loss import FocalLoss
import torch
import pandas as pd

def train(model, optimizer, exp_lr_scheduler, train_loader, dev_loader, args):
    history = pd.DataFrame()
    for epoch in range(args.epochs):
        train_model(model, optimizer, exp_lr_scheduler, epoch, train_loader, history, args.cuda)
        torch.save(model.state_dict(), args.model_path.format(epoch))
        history.to_csv(args.loss_path)
        # evaluate_model(model, epoch, dev_loader, history, args.cuda)

def train_model(model, optimizer, exp_lr_scheduler, epoch, train_loader, history=None, cuda=True):
    model.train()
    loss = 0
    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        if cuda:
            img_batch = img_batch.cuda()
            mask_batch = mask_batch.cuda()
            regr_batch = regr_batch.cuda()
        
        optimizer.zero_grad()
        output = model(img_batch)
        loss = FocalLoss(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
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
    
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            if cuda:
                img_batch = img_batch.cuda()
                mask_batch = mask_batch.cuda()
                regr_batch = regr_batch.cuda()

            output = model(img_batch)

            loss += FocalLoss(output, mask_batch, regr_batch, size_average=False).data
    loss /= len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
    
    print('Dev loss: {:.4f}'.format(loss))
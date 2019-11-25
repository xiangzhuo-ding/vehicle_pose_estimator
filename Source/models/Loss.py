import torch
from collections import defaultdict
from utils.preprocess import extract_coords
from utils.metrics import * 


def FocalLoss(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss

def EvaluationLoss(output, labels):
    output = output.data.cpu().numpy()

    # convert for the prediction from the model to the corrds format
    predictions = []

    for out in output:
        coords = extract_coords(out)
        predictions.append(coords)

    # we now need to com
    loss_dict = defaultdict(lambda: 0)
    for i in range(len(labels)):
        true_labels = np.array([float(x) for x in labels[i].split()])
        true_labels = true_labels.reshape(-1, 7)
        true_labels = true_labels[:, 1:]

        pred = []
        for x in predictions[i]:
            pred.append([x['yaw'], x['pitch'], x['roll'], x['x'], x['y'], x['z']])
        pred = np.array(pred)

       
 
        for true in true_labels:
            if len(pred) == 0:
                loss_dict['acc']  = 0 
                loss_dict['rot_loss'] = 180 / len(true_labels)
                p = true[3:].copy()
                p = [-x for x in p]
                loss_dict['distance_loss'] += (trans_dist(true[3:], p)) / len(true_labels)
                loss_dict['yaw_loss'] += abs(true[0]) / len(true_labels)
                loss_dict['pitch_loss'] += abs(true[1]) / len(true_labels)
                loss_dict['roll_loss'] += abs(true[2]) / len(true_labels)
                loss_dict['x_loss'] += abs(true[3]) / len(true_labels)
                loss_dict['y_loss'] += abs(true[4]) / len(true_labels)
                loss_dict['z_loss'] += abs(true[5]) / len(true_labels)
            else:
                acc_array = [get_acc(true, p) for p in pred]

                acc = max(acc_array)
                idx = acc_array.index(acc)

                loss_dict['acc'] += acc /len(true_labels)
                loss_dict['rot_loss'] += (rot_dist(true[:3], pred[idx][:3])) / len(true_labels)
                loss_dict['distance_loss'] += (trans_dist(true[3:], pred[idx][3:])) / len(true_labels)
                loss_dict['yaw_loss'] += abs(true[0] - pred[idx][0]) / len(true_labels)
                loss_dict['pitch_loss'] += abs(true[1] - pred[idx][1]) / len(true_labels)
                loss_dict['roll_loss'] += abs(true[2] - pred[idx][2]) / len(true_labels)
                loss_dict['x_loss'] += abs(true[3] - pred[idx][3]) / len(true_labels)
                loss_dict['y_loss'] += abs(true[4] - pred[idx][4]) / len(true_labels)
                loss_dict['z_loss'] += abs(true[5] - pred[idx][5]) / len(true_labels)

        return loss_dict

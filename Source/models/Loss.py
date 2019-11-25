import torch

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
z = np.array([float(x) for x in label[0].split()])
z = z.reshape(-1, 7)
z = z[:, 1:]



    pred = []
for x in predictions[0]:
    pred.append([x['yaw'], x['pitch'], x['roll'], x['x'], x['y'], x['z']])
pred = np.array(pred)
for true in z:
    distance_loss = min([trans_dist(true[3:], p[3:]) for p in pred])
    rotation_loss = min([rot_dist(true[:3], p[:3]) for p in pred])
    acc = max([get_acc(true, p) for p in pred])
    print(distance_loss, rotation_loss, acc)
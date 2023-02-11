import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pad_collate_reddit(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]

    lens = [len(x) for x in tweet]

    tweet = nn.utils.rnn.pad_sequence(tweet, batch_first=True, padding_value=0)

    target = torch.tensor(target)
    lens = torch.tensor(lens)

    return [target, tweet, lens]

def class_FScore(op, t, expt_type):
    FScores = []
    for i in range(expt_type):
        opc = op[t==i]
        tc = t[t==i]
        TP = (opc==tc).sum()
        FN = (tc>opc).sum()
        FP = (tc<opc).sum()

        GP = TP/(TP + FP  + 1e-8)
        GR = TP/(TP + FN + 1e-8)

        FS = 2 * GP * GR / (GP + GR + 1e-8)
        FScores.append(FS)
    return FScores

def gr_metrics(op, t):
    TP = (op==t).sum()
    FN = (t>op).sum()
    FP = (t<op).sum()

    GP = TP/(TP + FP)
    GR = TP/(TP + FN)

    FS = 2 * GP * GR / (GP + GR)

    OE = (t-op > 1).sum()
    OE = OE / op.shape[0]

    return GP, GR, FS, OE

def splits(df, dist_values):
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.sort_values(by='label').reset_index(drop=True)
    df_test = df[df['label']==0][0:dist_values[0]].reset_index(drop=True)
    for i in range(1,5):
        df_test = df_test.append(df[df['label']==i][0:dist_values[i]], ignore_index=True)

    for i in range(5):
        df.drop(df[df['label']==i].index[0:dist_values[i]], inplace=True)

    df = df.reset_index(drop=True)
    return df, df_test

def make_31(five_class):
    if five_class!=0:
        five_class=five_class-1
    return five_class


def CB_loss(logits, labels, samples_per_cls, no_of_classes, beta = 0.999, gamma = 2.0):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights, dtype=torch.float32).cuda()
    weights = weights.unsqueeze(0)
    
    print(weights)
    print(weights.repeat(labels_one_hot.shape[0], 1))
    
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)
    
    return weights

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def true_metric_loss(true, no_of_classes, scale=1):
    batch_size = true.size(0)
    true = true.view(batch_size,1)
    true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
    class_labels = torch.arange(no_of_classes).float().cuda()
    phi = (scale * torch.abs(class_labels - true_labels)).cuda()
    y = nn.Softmax(dim=1)(-phi)
    return y

def loss_function(output, labels, loss_type, expt_type, scale):
    if loss_type == 'OE':
        targets = true_metric_loss(labels, expt_type, scale)
        return torch.sum(- targets * F.log_softmax(output, -1), -1).mean()
    
    elif loss_type == 'focal':
        loss_fct = FocalLoss()
        loss = loss_fct(output,labels)
        return loss

    else:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(output, labels) 
        return loss



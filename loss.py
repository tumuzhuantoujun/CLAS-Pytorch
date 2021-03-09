import torch.nn as nn
import torch
import numpy as np

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape).cuda()
    result = result.scatter_(1, input, 1)
    return result

class oneclass_DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(oneclass_DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        assert predict.shape == target.shape, "predict & target shape don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(torch.pow(predict, 2) + torch.pow(target, 2), dim=1) + self.smooth
        loss = 1 - num / den
        return torch.mean(loss)

class hard_DiceLoss(nn.Module): # For images that have ground truth
    '''
    input:
        predict shape: batch_size * class_num * H * W
        target shape: batch_size * H * W
    '''
    def __init__(self, n_class = 4):
        super(hard_DiceLoss, self).__init__()
        self.loss = oneclass_DiceLoss()
        self.n_class = n_class

    def forward(self, predict, target):
        target = make_one_hot(target.unsqueeze(dim=1), self.n_class)
        assert predict.shape == target.shape, "predict & target shape don't match"
        total_loss = 0
        for i in range(target.shape[1]):
            total_loss += self.loss(predict[:, i], target[:, i])
        return total_loss / target.shape[1]

class soft_DiceLoss(nn.Module): # To keep consistency between cardiac segmentation and tracking
    '''
    input:
        predict1 & predict2 shape: batch_size * class_num * t * H * W
    '''
    def __init__(self):
        super(soft_DiceLoss, self).__init__()
        self.loss = oneclass_DiceLoss()

    def forward(self, predict1, predict2):
        assert predict1.shape == predict2.shape, "predict1 & predict2 shape don't match"
        total_loss = 0
        for i in range(predict1.shape[1]):
            total_loss += self.loss(predict1[:, i], predict2[:, i])
        return total_loss / predict1.shape[1]

def cross_correlation_loss(I, J, n=9):
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = torch.ones((1, channels, n, n))
    sum_filter = sum_filter.cuda()
    I_sum = torch.conv2d(I, sum_filter, padding = n // 2, stride=(1,1))
    J_sum = torch.conv2d(J, sum_filter,  padding = n // 2 ,stride=(1,1))
    I2_sum = torch.conv2d(I2, sum_filter, padding = n // 2, stride=(1,1))
    J2_sum = torch.conv2d(J2, sum_filter, padding = n // 2, stride=(1,1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding = n // 2, stride=(1,1))
    win_size = n**2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -torch.mean(cc)

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d/2.0

def vox_morph_cc_loss(flow, y, ytrue, n=9):
    cc = 0
    sm = 0
    for i in range(flow.shape[1]):
        cc = cc + cross_correlation_loss(ytrue[:,:,i], y[:,:,i], n)
        sm = sm + smooothing_loss(flow[:,i])
    return cc/flow.shape[1], sm/flow.shape[1]
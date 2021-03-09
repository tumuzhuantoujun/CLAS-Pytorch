import sys
import os
from optparse import OptionParser
import torch.backends.cudnn as cudnn
import torch
from torch import optim
from model import CLAS
from metric import *
from read_data import *
from registration_loss import *
from dice_loss import *
from output_data import *
from plot_curve import *
import datetime
import torch.nn as nn
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def adjust_learning_rate1(optimizer, lr):
    optimizer.param_groups[1]['lr'] = lr
def adjust_learning_rate_reg(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr

ch2 = np.load('./data/ch2.npy')
ch4 = np.load('./data/ch4.npy')
ch2_gt = np.load('./data/ch2_gt.npy')
ch4_gt = np.load('./data/ch4_gt.npy')

def train_net(net,
              epochs = 30,
              batch_size = 4,
              lr=1e-4,
              save_cp=True,
              gpu = True,
              fold = 1,
              ):
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, 800,
               100, str(save_cp), str(gpu)))

    f = open('./output/data/unet3D_seg_bireg_CV10/' + 'epoch_loss_log_down5_2_0.3r0.6t.txt', "a")
    print('Running {} fold'.format(fold), file=f)
    train, train_gt, val, val_gt = split_train_val_CV10(ch2, ch4, ch2_gt, ch4_gt, fold)
    val = torch.from_numpy(val)
    val_gt = torch.from_numpy(val_gt)
    criterion_CE = nn.CrossEntropyLoss()
    criterion_seg_Dice = DiceLoss()
    criterion_reg_Dice = DiceLoss_reg_label()
    epoch_loss_Dice = np.zeros((epochs,), dtype=np.float32)
    epoch_loss_CE = np.zeros((epochs,), dtype=np.float32)
    epoch_label_Dice_loss = np.zeros((epochs,), dtype=np.float32)
    epoch_Reg_EDES_loss = np.zeros((epochs,), dtype=np.float32)
    epoch_cc_loss = np.zeros((epochs,), dtype=np.float32)
    epoch_sm_loss = np.zeros((epochs,), dtype=np.float32)
    epoch_loss = np.zeros((epochs,), dtype=np.float32)
    #optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    reg_params = []
    reg_params += net.module.reg_warp.parameters()
    other_params = filter(lambda p: id(p) not in list(map(id, reg_params)), net.parameters())
    optimizer = optim.Adam([{'params': reg_params, 'lr': 0.5 * 1e-4}, {'params': other_params, 'lr': lr}],
                           weight_decay=0.0005)
    #torch.save(net.state_dict(), './CP_3Dunet_seg_bireg_CV10/' + 'CP0_fold{}_down5.pth'.format(fold))
    epochs_dice_ED_endo_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_dice_ED_epi_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_dice_ED_la_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_dice_ES_endo_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_dice_ES_epi_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)
    epochs_dice_ES_la_train = np.zeros((epochs, train.shape[0]), dtype=np.float32)

    epochs_dice_ED_endo_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_dice_ED_epi_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_dice_ED_la_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_dice_ES_endo_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_dice_ES_epi_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_dice_ES_la_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ED_endo_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ED_epi_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ED_la_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ES_endo_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ES_epi_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)
    epochs_hd_ES_la_val = np.zeros((epochs, val.shape[0]), dtype=np.float32)

    for epoch in range(epochs):
        state_train = np.random.get_state()
        np.random.shuffle(train)
        np.random.set_state(state_train)
        np.random.shuffle(train_gt)
        if epoch == 25:  # 25
            adjust_learning_rate1(optimizer, 1e-5)
            adjust_learning_rate_reg(optimizer, 1e-5)
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs), file=f)
        net.train()
        epoch_loss_tot = 0
        epoch_celoss_tot = 0
        epoch_diceloss_tot = 0
        epoch_label_Dice_loss_tot = 0
        epoch_reg_EDES_loss_tot = 0
        epoch_cc_loss_tot = 0
        epoch_sm_loss_tot = 0
        starttime = datetime.datetime.now()
        for batch_i in range(train.shape[0] // batch_size + 1):  #
            if (batch_i == train.shape[0] // batch_size):
                imgs = train[batch_i * batch_size: ]
                gt = train_gt[batch_i * batch_size: ]
                imgs = torch.from_numpy(imgs).unsqueeze(dim=1).cuda()
                gt = torch.from_numpy(gt).unsqueeze(dim=1).cuda()
            else:
                imgs = train[batch_i * batch_size: (batch_i + 1) * batch_size, ]
                gt = train_gt[batch_i * batch_size: (batch_i + 1) * batch_size, ]
                imgs = torch.from_numpy(imgs).unsqueeze(dim=1).cuda()
                gt = torch.from_numpy(gt).unsqueeze(dim=1).cuda()
            prob, reg_warp3d_forward, reg_warp3d_backward, flow_forward, flow_backward = net(imgs)
            # warp ori image
            gt_ED = make_one_hot(gt[:,:,0], 4)
            gt_ES = make_one_hot(gt[:,:,1], 4)
            # warp label forward across different class
            for i in range(9):
                if i == 0:
                    warp_class0 = F.grid_sample(gt_ED[:, 0].unsqueeze(1).float(), flow_forward[:, i])
                    warp_class1 = F.grid_sample(gt_ED[:, 1].unsqueeze(1).float(), flow_forward[:, i])
                    warp_class2 = F.grid_sample(gt_ED[:, 2].unsqueeze(1).float(), flow_forward[:, i])
                    warp_class3 = F.grid_sample(gt_ED[:, 3].unsqueeze(1).float(), flow_forward[:, i])
                    warp_label_forward = torch.cat([warp_class0, warp_class1, warp_class2, warp_class3], dim=1).unsqueeze(2)
                else :
                    warp_class0 = F.grid_sample(warp_label_forward[:, 0, i-1].unsqueeze(1).float(), flow_forward[:, i])
                    warp_class1 = F.grid_sample(warp_label_forward[:, 1, i-1].unsqueeze(1).float(), flow_forward[:, i])
                    warp_class2 = F.grid_sample(warp_label_forward[:, 2, i-1].unsqueeze(1).float(), flow_forward[:, i])
                    warp_class3 = F.grid_sample(warp_label_forward[:, 3, i-1].unsqueeze(1).float(), flow_forward[:, i])
                    warp = torch.cat([warp_class0, warp_class1, warp_class2, warp_class3], dim=1).unsqueeze(2)
                    warp_label_forward = torch.cat([warp_label_forward, warp], dim=2)
            # warp label backward across different class
            for i in range(9):
                if i == 0:
                    warp_class0 = F.grid_sample(gt_ES[:, 0].unsqueeze(1).float(), flow_backward[:, 8-i])
                    warp_class1 = F.grid_sample(gt_ES[:, 1].unsqueeze(1).float(), flow_backward[:, 8-i])
                    warp_class2 = F.grid_sample(gt_ES[:, 2].unsqueeze(1).float(), flow_backward[:, 8-i])
                    warp_class3 = F.grid_sample(gt_ES[:, 3].unsqueeze(1).float(), flow_backward[:, 8-i])
                    warp_label_backward = torch.cat([warp_class0, warp_class1, warp_class2, warp_class3], dim=1).unsqueeze(2)
                else :
                    warp_class0 = F.grid_sample(warp_label_backward[:, 0, i-1].unsqueeze(1).float(), flow_backward[:, 8-i])
                    warp_class1 = F.grid_sample(warp_label_backward[:, 1, i-1].unsqueeze(1).float(), flow_backward[:, 8-i])
                    warp_class2 = F.grid_sample(warp_label_backward[:, 2, i-1].unsqueeze(1).float(), flow_backward[:, 8-i])
                    warp_class3 = F.grid_sample(warp_label_backward[:, 3, i-1].unsqueeze(1).float(), flow_backward[:, 8-i])
                    warp = torch.cat([warp_class0, warp_class1, warp_class2, warp_class3], dim=1).unsqueeze(2)
                    warp_label_backward = torch.cat([warp_label_backward, warp], dim=2)
            for i in range(9):
                if i == 0:
                    warp_label_backward_rev = warp_label_backward[:,:,8-i].unsqueeze(dim=2)
                else:
                    warp_label_backward_rev = torch.cat([warp_label_backward_rev, warp_label_backward[:,:,8-i].unsqueeze(dim=2)],dim=2)
            warp_label_forward_b = torch.argmax(warp_label_forward, dim=1)
            warp_label_backward_rev_b = torch.argmax(warp_label_backward_rev, dim=1)

            Reg_middle_label_Dice_loss = (criterion_seg_Dice(prob[:,:,1:9], warp_label_forward_b[:,0:8]) + criterion_seg_Dice(prob[:,:,1:9], warp_label_backward_rev_b[:,1:]))/2
            Reg_EDES_loss = (criterion_seg_Dice(warp_label_forward[:,:,-1], gt[:, 0, 1]) + criterion_seg_Dice(warp_label_backward_rev[:,:,0], gt[:, 0, 0]))/2
            cc_loss_forward, sm_loss_forward = vox_morph_cc_loss(flow_forward, reg_warp3d_forward, imgs[:,:,1:])
            cc_loss_backward, sm_loss_backward = vox_morph_cc_loss(flow_backward, reg_warp3d_backward, imgs[:,:,0:-1])
            celoss = (criterion_CE(prob[:,:,0], gt[:, 0, 0]) + criterion_CE(prob[:,:,-1], gt[:, 0, -1])) / 2
            diceloss = (criterion_seg_Dice(prob[:,:,0], gt[:, 0, 0]) + criterion_seg_Dice(prob[:,:,-1], gt[:, 0, -1])) / 2
            cc_loss = (cc_loss_forward + cc_loss_backward) / 2
            sm_loss = (sm_loss_forward + sm_loss_backward) / 2
            if epoch < 10 :
                loss = celoss + diceloss + cc_loss + 10 * sm_loss
            else:
                loss = cc_loss + celoss + diceloss +  0.3 * Reg_middle_label_Dice_loss + 0.6 * Reg_EDES_loss + 10 * sm_loss

            print('Epoch : {}/{} --- batch : {}/{} --- Seg CE  -----  loss : {}'.format(epoch + 1, epochs, batch_i + 1, train.shape[0] // batch_size + 1, celoss.item()))
            print('Epoch : {}/{} --- batch : {}/{} --- Seg Dice ----- loss : {}'.format(epoch + 1, epochs, batch_i + 1, train.shape[0] // batch_size + 1, diceloss.item()))
            print('Epoch : {}/{} --- batch : {}/{} --- Reg CC  -----  loss : {}'.format(epoch + 1, epochs, batch_i + 1, train.shape[0] // batch_size + 1, cc_loss.item()))
            print('Epoch : {}/{} --- batch : {}/{} --- Reg Label Dice loss : {}'.format(epoch + 1, epochs, batch_i + 1, train.shape[0] // batch_size + 1, Reg_middle_label_Dice_loss.item()))
            print('Epoch : {}/{} --- batch : {}/{} --- Reg Label EDES loss : {}'.format(epoch + 1, epochs, batch_i + 1, train.shape[0] // batch_size + 1, Reg_EDES_loss.item()))
            print('Epoch : {}/{} --- batch : {}/{} --- Reg Warp SM -- loss : {}'.format(epoch + 1, epochs, batch_i + 1, train.shape[0] // batch_size + 1, sm_loss.item()))
            print('Epoch : {}/{} --- batch : {}/{} --- Tot loss : {}'.format(epoch + 1, epochs, batch_i + 1, train.shape[0] // batch_size + 1, loss.item()))
            epoch_loss_tot = epoch_loss_tot + loss.item()
            epoch_celoss_tot = epoch_celoss_tot + celoss.item()
            epoch_diceloss_tot = epoch_diceloss_tot + diceloss.item()
            epoch_label_Dice_loss_tot = epoch_label_Dice_loss_tot + Reg_middle_label_Dice_loss.item()
            epoch_reg_EDES_loss_tot = epoch_reg_EDES_loss_tot + Reg_EDES_loss.item()
            epoch_cc_loss_tot = epoch_cc_loss_tot + cc_loss.item()
            epoch_sm_loss_tot = epoch_sm_loss_tot + sm_loss.item()

            optimizer.zero_grad()  # 梯度置0
            loss.backward()  # 根据loss计算梯度
            optimizer.step()
            if sm_loss > 1 * 1e-4:
                nn.init.normal_(net.module.reg_warp.weight, mean=0, std=1e-5)
                nn.init.zeros_(net.module.reg_warp.bias)
        endtime = datetime.datetime.now()
        seconds = (endtime - starttime).seconds
        print('Fold {}'.format(fold))
        print('Epoch finished ! Seg CE  -----  loss : {}'.format(epoch_celoss_tot / (train.shape[0] // batch_size + 1)))
        print('Epoch finished ! Seg Dice ----- loss : {}'.format(epoch_diceloss_tot / (train.shape[0] // batch_size+ 1)))
        print('Epoch finished ! Reg CC  -----  loss : {}'.format(epoch_cc_loss_tot / (train.shape[0] // batch_size+ 1)))
        print('Epoch finished ! Reg Label Dice loss : {}'.format(epoch_label_Dice_loss_tot / (train.shape[0] // batch_size+ 1)))
        print('Epoch finished ! Reg Label EDES loss : {}'.format(epoch_reg_EDES_loss_tot / (train.shape[0] // batch_size+ 1)))
        print('Epoch finished ! Reg Warp SM -- loss : {}'.format(epoch_sm_loss_tot / (train.shape[0] // batch_size+ 1)))
        print('Epoch finished ! Tol Loss: {}'.format(epoch_loss_tot / (train.shape[0] // batch_size+ 1)))
        print('Epoch consume Time : {} s'.format(seconds))
        print('Fold {}'.format(fold), file=f)
        print('Epoch finished ! Seg CE  -----  loss : {}'.format(epoch_celoss_tot / (train.shape[0] // batch_size+ 1)), file=f)
        print('Epoch finished ! Seg Dice ----- loss : {}'.format(epoch_diceloss_tot / (train.shape[0] // batch_size+ 1)), file=f)
        print('Epoch finished ! Reg CC  -----  loss : {}'.format(epoch_cc_loss_tot / (train.shape[0] // batch_size+ 1)), file=f)
        print('Epoch finished ! Reg Label Dice loss : {}'.format(epoch_label_Dice_loss_tot / (train.shape[0] // batch_size+ 1)), file=f)
        print('Epoch finished ! Reg Label EDES loss : {}'.format(epoch_reg_EDES_loss_tot / (train.shape[0] // batch_size+ 1)), file=f)
        print('Epoch finished ! Reg Warp SM -- loss : {}'.format(epoch_sm_loss_tot / (train.shape[0] // batch_size+ 1)), file=f)
        print('Epoch finished ! Tol Loss: {}'.format(epoch_loss_tot / (train.shape[0] // batch_size+ 1)), file=f)
        print('Epoch consume Time : {} s'.format(seconds),file=f)

        epoch_loss_CE[epoch,] = epoch_celoss_tot / (train.shape[0] // batch_size + 1)
        epoch_loss_Dice[epoch,] = epoch_diceloss_tot / (train.shape[0] // batch_size+ 1)
        epoch_loss[epoch,] = epoch_loss_tot / (train.shape[0] // batch_size+ 1)
        epoch_label_Dice_loss[epoch,] = epoch_label_Dice_loss_tot / (train.shape[0] // batch_size+ 1)
        epoch_Reg_EDES_loss[epoch,] = epoch_reg_EDES_loss_tot / (train.shape[0] // batch_size+ 1)
        epoch_cc_loss[epoch,] = epoch_cc_loss_tot / (train.shape[0] // batch_size+ 1)
        epoch_sm_loss[epoch,] = epoch_sm_loss_tot / (train.shape[0] // batch_size+ 1)

        if 1:
            net.eval()
            # test validation set
            for batch_j in range(val.shape[0] // batch_size + 1):
                if (batch_j == val.shape[0] // batch_size):
                    imgs = val[batch_j * batch_size: ].unsqueeze(dim=1)
                    gt = val_gt[batch_j * batch_size: ].unsqueeze(dim=1)
                    imgs = imgs.cuda()
                    gt = gt.cuda()
                else:
                    imgs = val[batch_j * batch_size: (batch_j + 1) * batch_size, ].unsqueeze(dim=1)
                    gt = val_gt[batch_j * batch_size: (batch_j + 1) * batch_size, ].unsqueeze(dim=1)
                    imgs = imgs.cuda()
                    gt = gt.cuda()
                prob, _, _, _, _ = net(imgs)
                ED_endo, ED_epi, ED_LA, ES_endo, ES_epi, ES_LA = multi_class_dice(ED_seg = prob[:,:,0], ES_seg = prob[:,:,-1], gt = gt[:,0])
                if (batch_j == val.shape[0] // batch_size):
                    epochs_dice_ED_endo_val[epoch, batch_j * batch_size: ] = np.array(ED_endo)
                    epochs_dice_ED_epi_val[epoch, batch_j * batch_size: ] = np.array(ED_epi)
                    epochs_dice_ED_la_val[epoch, batch_j * batch_size: ] = np.array(ED_LA)
                    epochs_dice_ES_endo_val[epoch, batch_j * batch_size: ] = np.array(ES_endo)
                    epochs_dice_ES_epi_val[epoch, batch_j * batch_size: ] = np.array(ES_epi)
                    epochs_dice_ES_la_val[epoch, batch_j * batch_size: ] = np.array(ES_LA)
                else :
                    epochs_dice_ED_endo_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ED_endo)
                    epochs_dice_ED_epi_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ED_epi)
                    epochs_dice_ED_la_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ED_LA)
                    epochs_dice_ES_endo_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ES_endo)
                    epochs_dice_ES_epi_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ES_epi)
                    epochs_dice_ES_la_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(ES_LA)

                hd_ED_endo, hd_ED_epi, hd_ED_LA, hd_ES_endo, hd_ES_epi, hd_ES_LA = multi_class_hd(ED_seg=prob[:,:,0].cpu(),ES_seg=prob[:,:,-1].cpu(), gt=gt[:,0].cpu())
                if (batch_j == val.shape[0] // batch_size):
                    epochs_hd_ED_endo_val[epoch, batch_j * batch_size: ] = np.array(hd_ED_endo)
                    epochs_hd_ED_epi_val[epoch, batch_j * batch_size:] = np.array(hd_ED_epi)
                    epochs_hd_ED_la_val[epoch, batch_j * batch_size: ] = np.array(hd_ED_LA)
                    epochs_hd_ES_endo_val[epoch, batch_j * batch_size: ] = np.array(hd_ES_endo)
                    epochs_hd_ES_epi_val[epoch, batch_j * batch_size:] = np.array(hd_ES_epi)
                    epochs_hd_ES_la_val[epoch, batch_j * batch_size:] = np.array(hd_ES_LA)
                else:
                    epochs_hd_ED_endo_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ED_endo)
                    epochs_hd_ED_epi_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ED_epi)
                    epochs_hd_ED_la_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ED_LA)
                    epochs_hd_ES_endo_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ES_endo)
                    epochs_hd_ES_epi_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ES_epi)
                    epochs_hd_ES_la_val[epoch, batch_j * batch_size: (batch_j + 1) * batch_size] = np.array(hd_ES_LA)

            print('Validation Dice Coeff ED_endo: {} ± {} '.format(np.mean(epochs_dice_ED_endo_val[epoch,]), np.std(epochs_dice_ED_endo_val[epoch,])))
            print('Validation Dice Coeff ED_epi: {} ± {} '.format(np.mean(epochs_dice_ED_epi_val[epoch,]), np.std(epochs_dice_ED_epi_val[epoch,])))
            print('Validation Dice Coeff ED_LA: {} ± {} '.format(np.mean(epochs_dice_ED_la_val[epoch,]), np.std(epochs_dice_ED_la_val[epoch,])))
            print('Validation Dice Coeff ES_endo: {} ± {} '.format(np.mean(epochs_dice_ES_endo_val[epoch,]), np.std(epochs_dice_ES_endo_val[epoch,])))
            print('Validation Dice Coeff ES_epi: {} ± {} '.format(np.mean(epochs_dice_ES_epi_val[epoch,]), np.std(epochs_dice_ES_epi_val[epoch,])))
            print('Validation Dice Coeff ES_LA: {} ± {} '.format(np.mean(epochs_dice_ES_la_val[epoch,]), np.std(epochs_dice_ES_la_val[epoch,])))
            print('Validation HD ED_endo: {} ± {} '.format(np.mean(epochs_hd_ED_endo_val[epoch,]), np.std(epochs_hd_ED_endo_val[epoch,])))
            print('Validation HD ED_epi: {} ± {} '.format(np.mean(epochs_hd_ED_epi_val[epoch,]), np.std(epochs_hd_ED_epi_val[epoch,])))
            print('Validation HD ED_LA: {} ± {} '.format(np.mean(epochs_hd_ED_la_val[epoch,]), np.std(epochs_hd_ED_la_val[epoch,])))
            print('Validation HD ES_endo: {} ± {} '.format(np.mean(epochs_hd_ES_endo_val[epoch,]), np.std(epochs_hd_ES_endo_val[epoch,])))
            print('Validation HD ES_epi: {} ± {} '.format(np.mean(epochs_hd_ES_epi_val[epoch,]), np.std(epochs_hd_ES_epi_val[epoch,])))
            print('Validation HD ES_LA: {} ± {} '.format(np.mean(epochs_hd_ES_la_val[epoch,]), np.std(epochs_hd_ES_la_val[epoch,])))
            print('Validation Dice Coeff ED_endo: {} ± {} '.format(np.mean(epochs_dice_ED_endo_val[epoch,]), np.std(epochs_dice_ED_endo_val[epoch,])),file=f)
            print('Validation Dice Coeff ED_epi: {} ± {} '.format(np.mean(epochs_dice_ED_epi_val[epoch,]), np.std(epochs_dice_ED_epi_val[epoch,])),file=f)
            print('Validation Dice Coeff ED_LA: {} ± {} '.format(np.mean(epochs_dice_ED_la_val[epoch,]), np.std(epochs_dice_ED_la_val[epoch,])),file=f)
            print('Validation Dice Coeff ES_endo: {} ± {} '.format(np.mean(epochs_dice_ES_endo_val[epoch,]), np.std(epochs_dice_ES_endo_val[epoch,])),file=f)
            print('Validation Dice Coeff ES_epi: {} ± {} '.format(np.mean(epochs_dice_ES_epi_val[epoch,]), np.std(epochs_dice_ES_epi_val[epoch,])),file=f)
            print('Validation Dice Coeff ES_LA: {} ± {} '.format(np.mean(epochs_dice_ES_la_val[epoch,]), np.std(epochs_dice_ES_la_val[epoch,])),file=f)
            print('Validation HD ED_endo: {} ± {} '.format(np.mean(epochs_hd_ED_endo_val[epoch,]), np.std(epochs_hd_ED_endo_val[epoch,])),file=f)
            print('Validation HD ED_epi: {} ± {} '.format(np.mean(epochs_hd_ED_epi_val[epoch,]), np.std(epochs_hd_ED_epi_val[epoch,])),file=f)
            print('Validation HD ED_LA: {} ± {} '.format(np.mean(epochs_hd_ED_la_val[epoch,]), np.std(epochs_hd_ED_la_val[epoch,])),file=f)
            print('Validation HD ES_endo: {} ± {} '.format(np.mean(epochs_hd_ES_endo_val[epoch,]), np.std(epochs_hd_ES_endo_val[epoch,])),file=f)
            print('Validation HD ES_epi: {} ± {} '.format(np.mean(epochs_hd_ES_epi_val[epoch,]), np.std(epochs_hd_ES_epi_val[epoch,])),file=f)
            print('Validation HD ES_LA: {} ± {} '.format(np.mean(epochs_hd_ES_la_val[epoch,]), np.std(epochs_hd_ES_la_val[epoch,])),file=f)

        torch.save(net.state_dict(), './CP_3Dunet_seg_bireg_CV10/' + 'CP{}_fold{}_down5_0.3r0.6t.pth'.format(epoch + 1, fold))
        print('Checkpoint saved !')
    f.close()
    '''
    output_dice_hd_pat(
        [epochs_dice_ED_endo_train, epochs_dice_ED_epi_train, epochs_dice_ED_la_train, epochs_dice_ES_endo_train,
         epochs_dice_ES_epi_train, epochs_dice_ES_la_train, epochs_hd_ED_endo_train, epochs_hd_ED_epi_train,
         epochs_hd_ED_la_train, epochs_hd_ES_endo_train, epochs_hd_ES_epi_train, epochs_hd_ES_la_train],
        [epochs_dice_ED_endo_val, epochs_dice_ED_epi_val, epochs_dice_ED_la_val, epochs_dice_ES_endo_val,
         epochs_dice_ES_epi_val, epochs_dice_ES_la_val, epochs_hd_ED_endo_val, epochs_hd_ED_epi_val,
         epochs_hd_ED_la_val,epochs_hd_ES_endo_val, epochs_hd_ES_epi_val, epochs_hd_ES_la_val],
            './output/data/unet3D_seg_bireg_CV10/')
    '''
    seg_reg_plot_loss_dsc_curve([epoch_loss_CE,epoch_loss_Dice,epoch_cc_loss,epoch_label_Dice_loss,epoch_Reg_EDES_loss,epoch_sm_loss],
                        [np.mean(epochs_dice_ED_endo_val, axis=1), np.mean(epochs_dice_ED_epi_val,axis=1),
                         np.mean(epochs_dice_ED_la_val,axis=1), np.mean(epochs_dice_ES_endo_val, axis=1),
                         np.mean(epochs_dice_ES_epi_val,axis=1), np.mean(epochs_dice_ES_la_val,axis=1)],
                        [np.mean(epochs_hd_ED_endo_val, axis=1), np.mean(epochs_hd_ED_epi_val,axis=1),
                         np.mean(epochs_hd_ED_la_val,axis=1), np.mean(epochs_hd_ES_endo_val, axis=1),
                         np.mean(epochs_hd_ES_epi_val,axis=1), np.mean(epochs_hd_ES_la_val,axis=1)],
                        [np.mean(epochs_dice_ED_endo_train, axis=1), np.mean(epochs_dice_ED_epi_train,axis=1),
                         np.mean(epochs_dice_ED_la_train,axis=1), np.mean(epochs_dice_ES_endo_train, axis=1),
                         np.mean(epochs_dice_ES_epi_train,axis=1), np.mean(epochs_dice_ES_la_train,axis=1)],
                        './output/data/unet3D_seg_bireg_CV10/', fold)

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default = 30, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1E-4,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=1, help='downscaling factor of the images')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    net = CLAS()
    net = nn.DataParallel(net)
    net.cuda()
    cudnn.benchmark = True
    load_model = False
    for fold_k in range(1):
        if load_model:
            net.load_state_dict(torch.load('./CP/CP30_fold1.pth'))  # 用来加载模型参数
        try:
            train_net(net = net,
                      epochs = args.epochs,
                      batch_size = args.batchsize,
                      lr = args.lr,
                      gpu = args.gpu,
                      fold = fold_k + 1)
        except KeyboardInterrupt:
            print('Interrupt!')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

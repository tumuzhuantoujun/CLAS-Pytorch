import torch.nn.functional as F
import torch
import torch.nn as nn

class BasicBlock3D_seg_reg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock3D_seg_reg, self).__init__()
        self.convblock3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.convblock3d(x)
        return x

class BasicBlock3D_down_seg_reg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock3D_down_seg_reg, self).__init__()
        self.convblock3d = nn.Sequential(
            nn.MaxPool3d((1, 2, 2)),
            BasicBlock3D_seg_reg(in_channels, out_channels))
    def forward(self, x):
        x = self.convblock3d(x)
        return x

class bilinear_3dnetwork(nn.Module):
    def __init__(self):
        super(bilinear_3dnetwork, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, input):
        output = list()
        for i in range(input.shape[2]):
            output.append(self.up(input[:,:,i]))
        return torch.stack(output, dim=2)

class CLAS(nn.Module):
    def __init__(self, grid_shape=(256,256)):
        super(CLAS, self).__init__()
        self.conv0 = BasicBlock3D_seg_reg(1, 32)
        self.conv1 = BasicBlock3D_down_seg_reg(32, 64)
        self.conv2 = BasicBlock3D_down_seg_reg(64, 128)
        self.conv3 = BasicBlock3D_down_seg_reg(128, 256)
        self.conv4 = BasicBlock3D_down_seg_reg(256, 256)
        self.conv5 = BasicBlock3D_down_seg_reg(256, 256)
        self.up_conv5 = bilinear_3dnetwork()
        self.deconv1 = nn.Sequential(BasicBlock3D_seg_reg(256 + 256, 128),
                                     bilinear_3dnetwork())
        self.deconv2 = nn.Sequential(BasicBlock3D_seg_reg(128 + 256, 64),
                                     bilinear_3dnetwork())
        self.deconv3 = nn.Sequential(BasicBlock3D_seg_reg(64 + 128, 32),
                                     bilinear_3dnetwork())
        self.deconv4 = nn.Sequential(BasicBlock3D_seg_reg(32 + 64, 32),
                                     bilinear_3dnetwork())
        self.deconv5 = BasicBlock3D_seg_reg(32 + 32, 32)
        self.seg = nn.Conv3d(32, 4, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.reg = nn.Conv3d(32, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1))
        nn.init.normal_(self.reg.weight, mean=0, std=1e-6)
        nn.init.zeros_(self.reg.bias)
        grid = torch.meshgrid(torch.linspace(-1, 1, grid_shape[0]),
                              torch.linspace(-1, 1, grid_shape[1]))
        grid = torch.stack([grid[1], grid[0]], dim=-1)
        self.register_buffer("regular_grid", grid.unsqueeze(0))

    def forward(self, x):  # x: b * 1 * T * 256 * 256
        T = x.size(2)
        c0 = self.conv0(x)
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        d0 = self.up_conv5(c5)
        d1 = torch.cat((d0, c4), dim=1)
        d2 = self.deconv1(d1)
        d2 = torch.cat((d2, c3), dim=1)
        d3 = self.deconv2(d2)
        d3 = torch.cat((d3, c2), dim=1)
        d4 = self.deconv3(d3)
        d4 = torch.cat((d4, c1), dim=1)
        d5 = self.deconv4(d4)
        d5 = torch.cat((d5, c0), dim=1)
        d6 = self.deconv5(d5)
        seg_mask3d = self.seg(d6)
        reg_warp3d = self.reg(d6)
        reg_warp3d_forward = reg_warp3d[:,0:2]
        reg_warp3d_backward = reg_warp3d[:,2:]
        reg_warp3d_forward = reg_warp3d_forward.permute(0,2,3,4,1)
        reg_warp3d_backward = reg_warp3d_backward.permute(0,2,3,4,1)
        batch_regular_grid = self.regular_grid.repeat([x.size(0), 1, 1, 1])
        deformal_grid_forward, warp_forward, deformal_grid_backward, warp_backward = list(), list(), list(), list()
        # forward
        for i in range(T-1):
            deformal_grid_i = F.hardtanh(reg_warp3d_forward[:, i] + batch_regular_grid, -1, 1)
            deformal_grid_forward.append(deformal_grid_i)  # b * 256 * 256 * 2
            warp_forward.append(F.grid_sample(x[:, :, i], deformal_grid_i))  # b * 1 * 256 * 256
        # backward
        for i in range(T-1):
            deformal_grid_i = F.hardtanh(reg_warp3d_backward[:, i] + batch_regular_grid, -1, 1)
            deformal_grid_backward.append(deformal_grid_i)
            warp_backward.append(F.grid_sample(x[:, :, i+1], deformal_grid_i))
        return F.softmax(seg_mask3d, dim=1), \
               torch.stack(warp_forward, dim=2), torch.stack(warp_backward, dim=2), \
               torch.stack(deformal_grid_forward, dim=1), torch.stack(deformal_grid_backward, dim=1)
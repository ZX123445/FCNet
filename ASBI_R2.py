import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import nn
from lib.Res2Net_v1b import res2net50_v1b_26w_4s

# from lib.feature_1 import ETB #对一组图像中的随机一张图像进行特征提取   #消融实验
# from lib.feature1_wu_B import ETB #对一组图像中的随机一张图像进行特征提取   #消融实验

from lib.feature_1 import ETB #对一组图像中的随机一张图像进行特征提取     #原网络


from lib.feature_2 import sa_layer  #对一组图像进行协同特征提取
from lib.fusion import B  #融合模块

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size   , stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)

        self.sa = sa_layer(64)
        self.etb = ETB(64, 64)
        self.b = B()

        self.reduce4 = BasicConv2d(2048, 64, kernel_size=1)
        self.reduce3 = BasicConv2d(1024, 64, kernel_size=1)
        self.reduce2 = BasicConv2d(512, 64, kernel_size=1)
        self.reduce1 = BasicConv2d(256, 64, kernel_size=1)

        self.conv_192_64 = nn.Conv2d(192, 64, 1)
        self.conv_64_1 = nn.Conv2d(64, 1, 1)


    def forward(self, x):
        image_shape = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        r1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        r2 = self.resnet.layer2(r1)  # bs, 512, 44, 44
        r3 = self.resnet.layer3(r2)  # bs, 1024, 22, 22
        r4 = self.resnet.layer4(r3)  # bs, 2048, 11, 11

        r1 = self.reduce1(r1)   # r1[1, 64, 96, 96]
        r2 = self.reduce2(r2)   # r2[1, 64, 48, 48]
        r3 = self.reduce3(r3)   # r3[1, 64, 24, 24]
        r4 = self.reduce4(r4)   # r1[1, 64, 12, 12]

        r21 = F.interpolate(r2, r2.size()[2:], mode='bilinear', align_corners=False)
        r31 = F.interpolate(r3, r2.size()[2:], mode='bilinear', align_corners=False)
        r41 = F.interpolate(r4, r2.size()[2:], mode='bilinear', align_corners=False)

        """
        对一组图像进行特征提取
        对最高三层特征进行提取
        """
        f_fuse = torch.cat((r41, r31, r21), dim=1)
        f_fuse = self.conv_192_64(f_fuse)  #f_fuse[1, 64, 48, 48]
        f_fuse = self.sa(f_fuse)   #f_fuse[1, 64, 32, 32]


        """
        从一组输入图像中随机抽取一张进行协同特征提取
        对r2和r4层特征进行处理
        """
        f_24 = self.etb(r2, r4)


        """
        融合模块
        对一组图像的协同特征和一幅图像的特征进行融合
        """
        #判断协同特征和单幅图像的特征之间的维度是否相同

        if f_fuse.size()[2:] != f_24.size()[2:]:
            f_24 = F.interpolate(f_24, size=f_fuse.size()[2:], mode='bilinear', align_corners=False)
        # 此处为融合模块   对两个特征进行融合
        f = self.b(f_24, f_fuse)


        """
        需要对一组图像的的协同特征进行测试
        需要对一幅图像的特征进行测试
        """
        gt1 = self.conv_64_1(r1)
        gt2 = self.conv_64_1(r2)
        gt3 = self.conv_64_1(r3)

        # gt4 = self.conv_64_1(f_fuse)  #对协同特征进行预测
        # gt4 = self.conv_64_1(x_single) #对单幅图像特征进行预测
        gt4 = self.conv_64_1(f)  #对融合特征进行预测




        gt1 = F.interpolate(gt1, size=image_shape, mode='bilinear', align_corners=False)
        gt2 = F.interpolate(gt2, size=image_shape, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt3, size=image_shape, mode='bilinear', align_corners=False)
        gt4 = F.interpolate(gt4, size=image_shape, mode='bilinear', align_corners=False)

        return gt1, gt2, gt3, gt4



if __name__ =='__main__':
    from thop import profile
    net = Network().cuda()
    data = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(net, (data,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024*1024*1024), params / (1024*1024)))   #6.05G /  23M






# if __name__ == '__main__':
#     import numpy as np
#     from time import time
#
#     net = Network(imagenet_pretrained=False)
#     net.eval()
#
#     dump_x = torch.randn(1, 3, 384, 384)
#     frame_rate = np.zeros((1000, 1))
#     for i in range(1000):
#         start = time()
#         y = net(dump_x)
#         end = time()
#         running_frame_rate = 1 * float(1 / (end - start))
#         # print(i, '->', running_frame_rate)
#         frame_rate[i] = running_frame_rate
# print(np.mean(frame_rate))
# print(y.shape)
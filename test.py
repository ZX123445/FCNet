import torch
from thop import profile
import torch.nn.functional as F
import numpy as np
import os, argparse, cv2
from scipy import misc

from lib.ASBI_R2 import Network

from utils.data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--pth_path', type=str, default='./weight/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['test']:
    data_path = '/F:/CoCOD8K_1/{}/'.format(_data_name)
    save_path = './test_maps/{}/'.format(_data_name)
    model = Network(opt)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, group_name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        preds = model(image)
        res = preds[3]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {} '.format(_data_name, name))
        group_save_path = os.path.join(save_path, group_name)
        os.makedirs(group_save_path, exist_ok=True)
        save_path_with_group = os.path.join(group_save_path, name)
        cv2.imwrite(save_path_with_group, res*255)

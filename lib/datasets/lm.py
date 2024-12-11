# encoding: utf-8
'''
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: LineMOD.py
@time: 18-10-24 下午10:24
@desc: load LineMOD dataset
'''
from scipy.stats import truncnorm
import torch.utils.data as data
import numpy as np
import ref
import cv2
from utils.img import zoom_in, get_edges, xyxy_to_xywh, Crop_by_Pad
from utils.transform3d import prj_vtx_cam
from utils.io import read_pickle
import os, sys
from tqdm import tqdm
import utils.fancy_logger as logger
import pickle
from glob import glob 
import random 
from utils.eval import calc_rt_dist_m 
import matplotlib.pyplot as plt

np.set_printoptions(threshold = np.inf)

class LM(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.infos = self.load_lm_model_info(ref.lm_model_info_pth)
        self.cam_K = ref.K
        logger.info('==> initializing {} {} data.'.format(cfg.dataset.name, split))
        self.ger_fig = False
        self.c_fig = plt.figure()  # 生成一张画布
        self.c_fig.add_subplot(1, 1, 1)  # add_subplot在画布中添加一个axes(可以理解为子区域)，参数的前两个表示子区域的行列数，最后一个表示子区域的顺序
        plt.axis([-28, 92, -28, 92])
        plt.title("Distribution of Apes' Centers in ROI", x=0.5, y=-0.15)
        plt.plot([0, 64, 64, 0, 0], [0, 0, 64, 64, 0], 'r-', label='border', linewidth=1)
        # load dataset
        annot = []
        if split == 'test':
            cache_dir = os.path.join(ref.cache_dir, 'test/lm_test')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            for obj in tqdm(self.cfg.dataset.classes):
                cache_pth = os.path.join(cache_dir, '{}.npy'.format(obj))
                if not os.path.exists(cache_pth):
                    annot_cache = []
                    rgb_pths = glob(os.path.join(ref.lm_test_dir, obj, '*-color.png'))
                    for rgb_pth in tqdm(rgb_pths):
                        item = self.col_test_item(rgb_pth)
                        item['obj'] = obj
                        annot_cache.append(item)
                    np.save(cache_pth, annot_cache)
                annot.extend(np.load(cache_pth, allow_pickle=True).tolist())
            self.num = len(annot)
            self.annot = annot
            logger.info('load {} test samples.'.format(self.num))
        elif split == 'train':
            if 'real' in self.cfg.dataset.img_type:
                cache_dir = os.path.join(ref.cache_dir, 'train/real')
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                for obj in tqdm(self.cfg.dataset.classes):
                    cache_pth = os.path.join(cache_dir, '{}.npy'.format(obj))
                    if not os.path.exists(cache_pth):
                        annot_cache = []
                        rgb_pths = glob(os.path.join(ref.lm_train_real_dir, obj, '*-color.png'))
                        for rgb_pth in tqdm(rgb_pths):
                            item = self.col_train_item(rgb_pth)
                            item['obj'] = obj
                            annot_cache.append(item)
                        np.save(cache_pth, annot_cache)
                    annot.extend(np.load(cache_pth, allow_pickle=True).tolist())
                self.real_num = len(annot)
                logger.info('load {} real training samples.'.format(self.real_num))
            if 'imgn' in self.cfg.dataset.img_type:
                cache_dir = os.path.join(ref.cache_dir, 'train/imgn')
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                for obj in tqdm(self.cfg.dataset.classes):
                    cache_pth = os.path.join(cache_dir, '{}.npy'.format(obj))
                    if not os.path.exists(cache_pth):
                        annot_cache = []
                        coor_pths = sorted(glob(os.path.join(ref.lm_train_imgn_dir, obj, '*-coor.pkl')))
                        for coor_pth in tqdm(coor_pths):
                            item = self.col_imgn_item(coor_pth)
                            item['obj'] = obj
                            annot_cache.append(item)
                        np.save(cache_pth, annot_cache)
                    annot_obj = np.load(cache_pth, allow_pickle=True).tolist()
                    annot_obj_num = len(annot_obj)
                    if (annot_obj_num > self.cfg.dataset.syn_num) and (self.cfg.dataset.syn_samp_type != ''):
                        if self.cfg.dataset.syn_samp_type == 'uniform':
                            samp_idx = np.linspace(0, annot_obj_num - 1, self.cfg.dataset.syn_num, dtype=np.int32)
                        elif self.cfg.dataset.syn_samp_type == 'random':
                            samp_idx = random.sample(range(annot_obj_num), self.cfg.dataset.syn_num)
                        else:
                            raise ValueError
                        annot_obj = np.asarray(annot_obj)[samp_idx].tolist()
                    annot.extend(annot_obj)
                self.imgn_num = len(annot) - self.real_num
                logger.info('load {} imgn training samples.'.format(self.imgn_num))
            else:
                self.imgn_num = 0
            self.num = len(annot)
            self.annot = annot
            logger.info('load {} training samples, including {} real samples and {} synthetic samples.'.format(self.num, self.real_num, self.imgn_num))
            self.bg_list = self.load_bg_list()
        else:
            raise ValueError
        
    def col_test_item(self, rgb_pth):
        item = {}
        item['rgb_pth'] = rgb_pth
        item['pose'] = np.loadtxt(rgb_pth.replace('-color.png', '-pose.txt'))
        item['box'] = xyxy_to_xywh(np.loadtxt(rgb_pth.replace('-color.png', '-box_fasterrcnn.txt')))
        item['depth_pth'] = rgb_pth.replace('-color.png', '-depth.png')
        #item['mask_pth'] = rgb_pth.replace('-color.png', '-label.png')
        #item['coor_pth'] = rgb_pth.replace('-color.png', '-coor.pkl')
        #item['data_type'] = 'real'
        return item

    def col_train_item(self, rgb_pth):
        item = {}
        item['rgb_pth'] = rgb_pth
        item['pose'] = np.loadtxt(rgb_pth.replace('-color.png', '-pose.txt'))
        item['box'] = np.loadtxt(rgb_pth.replace('-color.png', '-box.txt'))
        item['mask_pth'] = rgb_pth.replace('-color.png', '-label.png')
        item['mask_visib_pth'] = rgb_pth.replace('-color.png', '-label_visib.png')
        item['new_mask_pth'] = rgb_pth.replace('-color.png', '-new_mask.png')
        item['new_mask_visib_pth'] = rgb_pth.replace('-color.png', '-new_mask_visib.png')
        if self.cfg.train.split_num == 2:  # 2
            item['mask_x1_pth'] = rgb_pth.replace('-color.png', '-mask_x1.png')
            item['mask_y1_pth'] = rgb_pth.replace('-color.png', '-mask_y1.png')
            item['mask_z1_pth'] = rgb_pth.replace('-color.png', '-mask_z1.png')
        if self.cfg.train.split_num == 3:  # 3
            item['mask_x1_N3_pth'] = rgb_pth.replace('-color.png', '-mask_x1_N3.png')
            item['mask_y1_N3_pth'] = rgb_pth.replace('-color.png', '-mask_y1_N3.png')
            item['mask_z1_N3_pth'] = rgb_pth.replace('-color.png', '-mask_z1_N3.png')
            item['mask_x2_N3_pth'] = rgb_pth.replace('-color.png', '-mask_x2_N3.png')
            item['mask_y2_N3_pth'] = rgb_pth.replace('-color.png', '-mask_y2_N3.png')
            item['mask_z2_N3_pth'] = rgb_pth.replace('-color.png', '-mask_z2_N3.png')
        if self.cfg.train.split_num == 4:  # 4
            item['mask_x1_pth'] = rgb_pth.replace('-color.png', '-mask_x1.png')
            item['mask_y1_pth'] = rgb_pth.replace('-color.png', '-mask_y1.png')
            item['mask_z1_pth'] = rgb_pth.replace('-color.png', '-mask_z1.png')
            item['mask_x2_pth'] = rgb_pth.replace('-color.png', '-mask_x2.png')
            item['mask_y2_pth'] = rgb_pth.replace('-color.png', '-mask_y2.png')
            item['mask_z2_pth'] = rgb_pth.replace('-color.png', '-mask_z2.png')
            item['mask_x2_nb_pth'] = rgb_pth.replace('-color.png', '-mask_x2_nb.png')
            item['mask_y2_nb_pth'] = rgb_pth.replace('-color.png', '-mask_y2_nb.png')
            item['mask_z2_nb_pth'] = rgb_pth.replace('-color.png', '-mask_z2_nb.png')
        if self.cfg.train.split_num == 5:  # 5
            item['mask_x1_N5_pth'] = rgb_pth.replace('-color.png', '-mask_x1_N5.png')
            item['mask_y1_N5_pth'] = rgb_pth.replace('-color.png', '-mask_y1_N5.png')
            item['mask_z1_N5_pth'] = rgb_pth.replace('-color.png', '-mask_z1_N5.png')
            item['mask_x2_N5_pth'] = rgb_pth.replace('-color.png', '-mask_x2_N5.png')
            item['mask_y2_N5_pth'] = rgb_pth.replace('-color.png', '-mask_y2_N5.png')
            item['mask_z2_N5_pth'] = rgb_pth.replace('-color.png', '-mask_z2_N5.png')
            item['mask_x3_N5_pth'] = rgb_pth.replace('-color.png', '-mask_x3_N5.png')
            item['mask_y3_N5_pth'] = rgb_pth.replace('-color.png', '-mask_y3_N5.png')
            item['mask_z3_N5_pth'] = rgb_pth.replace('-color.png', '-mask_z3_N5.png')
        if self.cfg.train.split_num == 6:  # 6
            item['mask_x1_N6_pth'] = rgb_pth.replace('-color.png', '-mask_x1_N6.png')
            item['mask_y1_N6_pth'] = rgb_pth.replace('-color.png', '-mask_y1_N6.png')
            item['mask_z1_N6_pth'] = rgb_pth.replace('-color.png', '-mask_z1_N6.png')
            item['mask_x2_N6_pth'] = rgb_pth.replace('-color.png', '-mask_x2_N6.png')
            item['mask_y2_N6_pth'] = rgb_pth.replace('-color.png', '-mask_y2_N6.png')
            item['mask_z2_N6_pth'] = rgb_pth.replace('-color.png', '-mask_z2_N6.png')
            item['mask_x3_N6_pth'] = rgb_pth.replace('-color.png', '-mask_x3_N6.png')
            item['mask_y3_N6_pth'] = rgb_pth.replace('-color.png', '-mask_y3_N6.png')
            item['mask_z3_N6_pth'] = rgb_pth.replace('-color.png', '-mask_z3_N6.png')
        if self.cfg.train.split_num == 7:  # 7
            item['mask_x1_N7_pth'] = rgb_pth.replace('-color.png', '-mask_x1_N7.png')
            item['mask_y1_N7_pth'] = rgb_pth.replace('-color.png', '-mask_y1_N7.png')
            item['mask_z1_N7_pth'] = rgb_pth.replace('-color.png', '-mask_z1_N7.png')
            item['mask_x2_N7_pth'] = rgb_pth.replace('-color.png', '-mask_x2_N7.png')
            item['mask_y2_N7_pth'] = rgb_pth.replace('-color.png', '-mask_y2_N7.png')
            item['mask_z2_N7_pth'] = rgb_pth.replace('-color.png', '-mask_z2_N7.png')
            item['mask_x3_N7_pth'] = rgb_pth.replace('-color.png', '-mask_x3_N7.png')
            item['mask_y3_N7_pth'] = rgb_pth.replace('-color.png', '-mask_y3_N7.png')
            item['mask_z3_N7_pth'] = rgb_pth.replace('-color.png', '-mask_z3_N7.png')
        if self.cfg.train.split_num == 8:  # 8
            item['mask_x1_pth'] = rgb_pth.replace('-color.png', '-mask_x1.png')
            item['mask_y1_pth'] = rgb_pth.replace('-color.png', '-mask_y1.png')
            item['mask_z1_pth'] = rgb_pth.replace('-color.png', '-mask_z1.png')
            item['mask_x2_pth'] = rgb_pth.replace('-color.png', '-mask_x2.png')
            item['mask_y2_pth'] = rgb_pth.replace('-color.png', '-mask_y2.png')
            item['mask_z2_pth'] = rgb_pth.replace('-color.png', '-mask_z2.png')
            item['mask_x3_pth'] = rgb_pth.replace('-color.png', '-mask_x3.png')
            item['mask_y3_pth'] = rgb_pth.replace('-color.png', '-mask_y3.png')
            item['mask_z3_pth'] = rgb_pth.replace('-color.png', '-mask_z3.png')
        item['coor_pth'] = rgb_pth.replace('-color.png', '-coor.pkl')
        item['data_type'] = 'real'
        return item

    def col_imgn_item(self, coor_pth):
        item = {}
        item['coor_pth'] = coor_pth
        item['rgb_pth'] = coor_pth.replace('-coor.pkl', '-color.png')
        item['pose'] = np.loadtxt(coor_pth.replace('-coor.pkl', '-pose.txt'))
        item['box'] = np.loadtxt(coor_pth.replace('-coor.pkl', '-box.txt'))
        item['mask_pth'] = coor_pth.replace('-coor.pkl', '-label.png')
        item['new_mask_pth'] = coor_pth.replace('-coor.pkl', '-new_mask.png')
        item['new_mask_visib_pth'] = coor_pth.replace('-coor.pkl', '-new_mask_visib.png')
        if self.cfg.train.split_num == 2:  # 2
            item['mask_x1_pth'] = coor_pth.replace('-coor.pkl', '-mask_x1.png')
            item['mask_y1_pth'] = coor_pth.replace('-coor.pkl', '-mask_y1.png')
            item['mask_z1_pth'] = coor_pth.replace('-coor.pkl', '-mask_z1.png')
        if self.cfg.train.split_num == 3:  # 3
            item['mask_x1_N3_pth'] = coor_pth.replace('-coor.pkl', '-mask_x1_N3.png')
            item['mask_y1_N3_pth'] = coor_pth.replace('-coor.pkl', '-mask_y1_N3.png')
            item['mask_z1_N3_pth'] = coor_pth.replace('-coor.pkl', '-mask_z1_N3.png')
            item['mask_x2_N3_pth'] = coor_pth.replace('-coor.pkl', '-mask_x2_N3.png')
            item['mask_y2_N3_pth'] = coor_pth.replace('-coor.pkl', '-mask_y2_N3.png')
            item['mask_z2_N3_pth'] = coor_pth.replace('-coor.pkl', '-mask_z2_N3.png')
        if self.cfg.train.split_num == 4:  # 4
            item['mask_x1_pth'] = coor_pth.replace('-coor.pkl', '-mask_x1.png')
            item['mask_y1_pth'] = coor_pth.replace('-coor.pkl', '-mask_y1.png')
            item['mask_z1_pth'] = coor_pth.replace('-coor.pkl', '-mask_z1.png')
            item['mask_x2_pth'] = coor_pth.replace('-coor.pkl', '-mask_x2.png')
            item['mask_y2_pth'] = coor_pth.replace('-coor.pkl', '-mask_y2.png')
            item['mask_z2_pth'] = coor_pth.replace('-coor.pkl', '-mask_z2.png')
            item['mask_x2_nb_pth'] = coor_pth.replace('-coor.pkl', '-mask_x2_nb.png')
            item['mask_y2_nb_pth'] = coor_pth.replace('-coor.pkl', '-mask_y2_nb.png')
            item['mask_z2_nb_pth'] = coor_pth.replace('-coor.pkl', '-mask_z2_nb.png')
        if self.cfg.train.split_num == 5:  # 5
            item['mask_x1_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_x1_N5.png')
            item['mask_y1_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_y1_N5.png')
            item['mask_z1_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_z1_N5.png')
            item['mask_x2_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_x2_N5.png')
            item['mask_y2_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_y2_N5.png')
            item['mask_z2_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_z2_N5.png')
            item['mask_x3_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_x3_N5.png')
            item['mask_y3_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_y3_N5.png')
            item['mask_z3_N5_pth'] = coor_pth.replace('-coor.pkl', '-mask_z3_N5.png')
        if self.cfg.train.split_num == 6:  # 6
            item['mask_x1_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_x1_N6.png')
            item['mask_y1_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_y1_N6.png')
            item['mask_z1_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_z1_N6.png')
            item['mask_x2_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_x2_N6.png')
            item['mask_y2_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_y2_N6.png')
            item['mask_z2_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_z2_N6.png')
            item['mask_x3_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_x3_N6.png')
            item['mask_y3_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_y3_N6.png')
            item['mask_z3_N6_pth'] = coor_pth.replace('-coor.pkl', '-mask_z3_N6.png')
        if self.cfg.train.split_num == 7:  # 7
            item['mask_x1_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_x1_N7.png')
            item['mask_y1_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_y1_N7.png')
            item['mask_z1_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_z1_N7.png')
            item['mask_x2_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_x2_N7.png')
            item['mask_y2_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_y2_N7.png')
            item['mask_z2_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_z2_N7.png')
            item['mask_x3_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_x3_N7.png')
            item['mask_y3_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_y3_N7.png')
            item['mask_z3_N7_pth'] = coor_pth.replace('-coor.pkl', '-mask_z3_N7.png')
        if self.cfg.train.split_num == 8:  # 8
            item['mask_x1_pth'] = coor_pth.replace('-coor.pkl', '-mask_x1.png')
            item['mask_y1_pth'] = coor_pth.replace('-coor.pkl', '-mask_y1.png')
            item['mask_z1_pth'] = coor_pth.replace('-coor.pkl', '-mask_z1.png')
            item['mask_x2_pth'] = coor_pth.replace('-coor.pkl', '-mask_x2.png')
            item['mask_y2_pth'] = coor_pth.replace('-coor.pkl', '-mask_y2.png')
            item['mask_z2_pth'] = coor_pth.replace('-coor.pkl', '-mask_z2.png')
            item['mask_x3_pth'] = coor_pth.replace('-coor.pkl', '-mask_x3.png')
            item['mask_y3_pth'] = coor_pth.replace('-coor.pkl', '-mask_y3.png')
            item['mask_z3_pth'] = coor_pth.replace('-coor.pkl', '-mask_z3.png')
        item['data_type'] = 'imgn'
        return item

    @staticmethod
    def load_lm_model_info(info_pth):
        infos = {}
        with open(info_pth, 'r') as f:
            for line in f.readlines():
                items = line.strip().split(' ')
                cls_idx = int(items[0])
                infos[cls_idx] = {}
                infos[cls_idx]['diameter'] = float(items[2]) / 1000. # unit: mm => m
                infos[cls_idx]['min_x'] = float(items[4]) / 1000.
                infos[cls_idx]['min_y'] = float(items[6]) / 1000.
                infos[cls_idx]['min_z'] = float(items[8]) / 1000.
        return infos

    @staticmethod
    def load_bg_list():
        path = os.path.join(ref.bg_dir, 'VOC2012/ImageSets/Main/diningtable_trainval.txt')
        with open(path, 'r') as f:
            bg_list = [line.strip('\r\n').split()[0] for line in f.readlines() if
                                line.strip('\r\n').split()[1] == '1']
        return bg_list

    @staticmethod
    def load_bg_im(im_real, bg_list):
        h, w, c = im_real.shape
        bg_num = len(bg_list)
        idx = random.randint(0, bg_num - 1)
        bg_path = os.path.join(ref.bg_dir, 'VOC2012/JPEGImages/{}.jpg'.format(bg_list[idx]))
        bg_im = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        bg_h, bg_w, bg_c = bg_im.shape
        real_hw_ratio = float(h) / float(w)
        bg_hw_ratio = float(bg_h) / float(bg_w)
        if real_hw_ratio <= bg_hw_ratio:
            crop_w = bg_w
            crop_h = int(bg_w * real_hw_ratio)
        else:
            crop_h = bg_h 
            crop_w = int(bg_h / bg_hw_ratio)
        bg_im = bg_im[:crop_h, :crop_w, :]
        bg_im = cv2.resize(bg_im, (w, h), interpolation=cv2.INTER_LINEAR)
        return bg_im
        
    def change_bg(self, rgb, msk):
        """
        change image's background
        """
        bg_im = self.load_bg_im(rgb, self.bg_list)
        msk = np.dstack([msk, msk, msk]).astype(np.bool)
        bg_im[msk] = rgb[msk]
        return bg_im

    def load_obj(self, idx):
        return self.annot[idx]['obj']

    def load_type(self, idx):
        return self.annot[idx]['data_type']

    def load_pose(self, idx):
        return self.annot[idx]['pose']

    def load_box(self, idx):
        return self.annot[idx]['box']

    def load_msk(self, idx):
        mask = cv2.imread(self.annot[idx]['mask_pth'], cv2.IMREAD_GRAYSCALE)
        return mask

    def load_msk_visib(self, idx):
        mask = cv2.imread(self.annot[idx]['mask_pth'], cv2.IMREAD_GRAYSCALE)
        return mask

    def load_new_msk(self, idx):
        new_mask = cv2.imread(self.annot[idx]['new_mask_pth'], cv2.IMREAD_GRAYSCALE)
        return new_mask

    def load_msk_x1(self, idx):
        mask_x1 = cv2.imread(self.annot[idx]['mask_x1_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x1

    def load_msk_y1(self, idx):
        mask_y1 = cv2.imread(self.annot[idx]['mask_y1_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y1

    def load_msk_z1(self, idx):
        mask_z1 = cv2.imread(self.annot[idx]['mask_z1_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z1

    def load_msk_x2(self, idx):
        mask_x2 = cv2.imread(self.annot[idx]['mask_x2_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x2

    def load_msk_y2(self, idx):
        mask_y2 = cv2.imread(self.annot[idx]['mask_y2_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y2

    def load_msk_z2(self, idx):
        mask_z2 = cv2.imread(self.annot[idx]['mask_z2_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z2

    def load_msk_x2_nb(self, idx):
        mask_x2_nb = cv2.imread(self.annot[idx]['mask_x2_nb_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x2_nb

    def load_msk_y2_nb(self, idx):
        mask_y2_nb = cv2.imread(self.annot[idx]['mask_y2_nb_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y2_nb

    def load_msk_z2_nb(self, idx):
        mask_z2_nb = cv2.imread(self.annot[idx]['mask_z2_nb_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z2_nb

    def load_msk_x3(self, idx):
        mask_x3 = cv2.imread(self.annot[idx]['mask_x3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x3

    def load_msk_y3(self, idx):
        mask_y3 = cv2.imread(self.annot[idx]['mask_y3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y3

    def load_msk_z3(self, idx):
        mask_z3 = cv2.imread(self.annot[idx]['mask_z3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z3

    def load_msk_x1_N3(self, idx):
        mask_x1_N3 = cv2.imread(self.annot[idx]['mask_x1_N3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x1_N3

    def load_msk_x2_N3(self, idx):
        mask_x2_N3 = cv2.imread(self.annot[idx]['mask_x2_N3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x2_N3

    def load_msk_y1_N3(self, idx):
        mask_y1_N3 = cv2.imread(self.annot[idx]['mask_y1_N3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y1_N3

    def load_msk_y2_N3(self, idx):
        mask_y2_N3 = cv2.imread(self.annot[idx]['mask_y2_N3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y2_N3

    def load_msk_z1_N3(self, idx):
        mask_z1_N3 = cv2.imread(self.annot[idx]['mask_z1_N3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z1_N3

    def load_msk_z2_N3(self, idx):
        mask_z2_N3 = cv2.imread(self.annot[idx]['mask_z2_N3_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z2_N3

    def load_msk_x1_N5(self, idx):
        mask_x1_N5 = cv2.imread(self.annot[idx]['mask_x1_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x1_N5

    def load_msk_x2_N5(self, idx):
        mask_x2_N5 = cv2.imread(self.annot[idx]['mask_x2_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x2_N5

    def load_msk_x3_N5(self, idx):
        mask_x3_N5 = cv2.imread(self.annot[idx]['mask_x3_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x3_N5

    def load_msk_y1_N5(self, idx):
        mask_y1_N5 = cv2.imread(self.annot[idx]['mask_y1_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y1_N5

    def load_msk_y2_N5(self, idx):
        mask_y2_N5 = cv2.imread(self.annot[idx]['mask_y2_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y2_N5

    def load_msk_y3_N5(self, idx):
        mask_y3_N5 = cv2.imread(self.annot[idx]['mask_y3_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y3_N5

    def load_msk_z1_N5(self, idx):
        mask_z1_N5 = cv2.imread(self.annot[idx]['mask_z1_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z1_N5

    def load_msk_z2_N5(self, idx):
        mask_z2_N5 = cv2.imread(self.annot[idx]['mask_z2_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z2_N5

    def load_msk_z3_N5(self, idx):
        mask_z3_N5 = cv2.imread(self.annot[idx]['mask_z3_N5_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z3_N5

    def load_msk_x1_N6(self, idx):
        mask_x1_N6 = cv2.imread(self.annot[idx]['mask_x1_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x1_N6

    def load_msk_x2_N6(self, idx):
        mask_x2_N6 = cv2.imread(self.annot[idx]['mask_x2_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x2_N6

    def load_msk_x3_N6(self, idx):
        mask_x3_N6 = cv2.imread(self.annot[idx]['mask_x3_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x3_N6

    def load_msk_y1_N6(self, idx):
        mask_y1_N6 = cv2.imread(self.annot[idx]['mask_y1_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y1_N6

    def load_msk_y2_N6(self, idx):
        mask_y2_N6 = cv2.imread(self.annot[idx]['mask_y2_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y2_N6

    def load_msk_y3_N6(self, idx):
        mask_y3_N6 = cv2.imread(self.annot[idx]['mask_y3_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y3_N6

    def load_msk_z1_N6(self, idx):
        mask_z1_N6 = cv2.imread(self.annot[idx]['mask_z1_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z1_N6

    def load_msk_z2_N6(self, idx):
        mask_z2_N6 = cv2.imread(self.annot[idx]['mask_z2_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z2_N6

    def load_msk_z3_N6(self, idx):
        mask_z3_N6 = cv2.imread(self.annot[idx]['mask_z3_N6_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z3_N6

    def load_msk_x1_N7(self, idx):
        mask_x1_N7 = cv2.imread(self.annot[idx]['mask_x1_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x1_N7

    def load_msk_x2_N7(self, idx):
        mask_x2_N7 = cv2.imread(self.annot[idx]['mask_x2_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x2_N7

    def load_msk_x3_N7(self, idx):
        mask_x3_N7 = cv2.imread(self.annot[idx]['mask_x3_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_x3_N7

    def load_msk_y1_N7(self, idx):
        mask_y1_N7 = cv2.imread(self.annot[idx]['mask_y1_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y1_N7

    def load_msk_y2_N7(self, idx):
        mask_y2_N7 = cv2.imread(self.annot[idx]['mask_y2_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y2_N7

    def load_msk_y3_N7(self, idx):
        mask_y3_N7 = cv2.imread(self.annot[idx]['mask_y3_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_y3_N7

    def load_msk_z1_N7(self, idx):
        mask_z1_N7 = cv2.imread(self.annot[idx]['mask_z1_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z1_N7

    def load_msk_z2_N7(self, idx):
        mask_z2_N7 = cv2.imread(self.annot[idx]['mask_z2_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z2_N7

    def load_msk_z3_N7(self, idx):
        mask_z3_N7 = cv2.imread(self.annot[idx]['mask_z3_N7_pth'], cv2.IMREAD_GRAYSCALE)
        return mask_z3_N7

    def load_rgb(self, idx):
        return cv2.imread(self.annot[idx]['rgb_pth'])

    def load_coor(self, idx, restore=True, coor_h=480, coor_w=640):         # the usage of pickle!
        try:
            coor_load = read_pickle(self.annot[idx]['coor_pth'])
        except:
            print('coor_pth: {}'.format(self.annot[idx]['coor_pth']))
            raise
        if not restore:
            return coor_load['coor']
        else:
            u = coor_load['u']
            l = coor_load['l']
            h = coor_load['h']
            w = coor_load['w']
            coor = np.zeros((coor_h, coor_w, 3)).astype(np.float32)
            coor[u:(u+h),l:(l+w),:] = coor_load['coor']
            return coor

    def xywh_to_cs_dzi(self, xywh, s_ratio, s_max=None, tp='uniform'):
        x, y, w, h = xywh
        if tp == 'gaussian':
            sigma = 1
            shift = truncnorm.rvs(-self.cfg.augment.shift_ratio / sigma, self.cfg.augment.shift_ratio / sigma, scale=sigma, size=2)
            scale = 1+truncnorm.rvs(-self.cfg.augment.scale_ratio / sigma, self.cfg.augment.scale_ratio / sigma, scale=sigma, size=1)
        elif tp == 'uniform':
            scale = 1+self.cfg.augment.scale_ratio * (2*np.random.random_sample()-1)
            shift = self.cfg.augment.shift_ratio * (2*np.random.random_sample(2)-1)
        else:
            raise
        c = np.array([x+w*(0.5+shift[1]), y+h*(0.5+shift[0])]) # [c_w, c_h]
        s = max(w, h)*s_ratio*scale
        if s_max != None:
            s = min(s, s_max)
        return c, s

    @staticmethod
    def xywh_to_cs(xywh, s_ratio, s_max=None):
        x, y, w, h = xywh
        c = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        s = max(w, h)*s_ratio
        if s_max != None:
            s = min(s, s_max)
        return c, s

    def denoise_coor(self, coor):
        """
        denoise coordinates by median blur
        """
        coor_blur = cv2.medianBlur(coor, 3)
        edges = get_edges(coor)
        coor[edges != 0] = coor_blur[edges != 0]
        return coor

    def norm_coor(self, coor, obj_id):
        """
        normalize coordinates by object size originally
        """
        coor[:, :, 0] = coor[:, :, 0] / abs(self.infos[obj_id]['min_x'])
        coor[:, :, 1] = coor[:, :, 1] / abs(self.infos[obj_id]['min_y'])
        coor[:, :, 2] = coor[:, :, 2] / abs(self.infos[obj_id]['min_z'])

        return coor

    def norm_coor_2(self, coor, obj_id):
        """
        normalize coordinates by object size in condition of spliting ply into 8 parts
        """
        coor[:, :, 0] = coor[:, :, 0] / abs(self.infos[obj_id]['min_x'])
        coor[:, :, 1] = coor[:, :, 1] / abs(self.infos[obj_id]['min_y'])
        coor[:, :, 2] = coor[:, :, 2] / abs(self.infos[obj_id]['min_z'])
        #for x in coor_x.shape[]
        for i in range(3):
            for x in range(64):
                for y in range(64):
                    if coor[y, x, i] <= 0:
                        coor[y, x, i] = -2 * coor[y, x, i] - 1
                    else:
                        coor[y, x, i] = 2 * coor[y, x, i] - 1

        return coor

    def norm_coor_3(self, coor, obj_id):
        """
        normalize coordinates by object size in condition of spliting ply into 27 parts
        """
        coor[:, :, 0] = coor[:, :, 0] / abs(self.infos[obj_id]['min_x'])
        coor[:, :, 1] = coor[:, :, 1] / abs(self.infos[obj_id]['min_y'])
        coor[:, :, 2] = coor[:, :, 2] / abs(self.infos[obj_id]['min_z'])
        #for x in coor_x.shape[]
        for i in range(3):
            for x in range(64):
                for y in range(64):
                    if coor[y, x, i] <= -1/3:
                        coor[y, x, i] = -3 * coor[y, x, i] - 2
                    elif coor[y, x, i] > -1/3 and coor[y, x, i] <= 1/3:
                        coor[y, x, i] = 3 * coor[y, x, i]
                    else:
                        coor[y, x, i] = -3 * coor[y, x, i] + 2

        return coor

    def norm_coor_4(self, coor, obj_id):
        """
        normalize coordinates by object size in condition of spliting ply into 64 parts
        """
        coor[:, :, 0] = coor[:, :, 0] / abs(self.infos[obj_id]['min_x'])
        coor[:, :, 1] = coor[:, :, 1] / abs(self.infos[obj_id]['min_y'])
        coor[:, :, 2] = coor[:, :, 2] / abs(self.infos[obj_id]['min_z'])
        # for x in coor_x.shape[]
        for i in range(3):
            for x in range(64):
                for y in range(64):
                    if coor[y, x, i] <= -0.5:
                        coor[y, x, i] = -4 * coor[y, x, i] - 3
                    elif coor[y, x, i] > -0.5 and coor[y, x, i] <= 0:
                        coor[y, x, i] = 4 * coor[y, x, i] + 1
                    elif coor[y, x, i] > 0 and coor[y, x, i] <= 0.5:
                        coor[y, x, i] = -4 * coor[y, x, i] + 1
                    else:
                        coor[y, x, i] = 4 * coor[y, x, i] - 3

        return coor

    def norm_coor_5(self, coor, obj_id):
        """
        normalize coordinates by object size in condition of spliting ply into 125 parts
        """
        coor[:, :, 0] = coor[:, :, 0] / abs(self.infos[obj_id]['min_x'])
        coor[:, :, 1] = coor[:, :, 1] / abs(self.infos[obj_id]['min_y'])
        coor[:, :, 2] = coor[:, :, 2] / abs(self.infos[obj_id]['min_z'])
        # for x in coor_x.shape[]
        for i in range(3):
            for x in range(64):
                for y in range(64):
                    if coor[y, x, i] <= -3/5:
                        coor[y, x, i] = -5 * coor[y, x, i] - 4
                    elif coor[y, x, i] > -3/5 and coor[y, x, i] <= -1/5:
                        coor[y, x, i] = 5 * coor[y, x, i] + 2
                    elif coor[y, x, i] > -1/5 and coor[y, x, i] <= 1/5:
                        coor[y, x, i] = -5 * coor[y, x, i]
                    elif coor[y, x, i] > 1/5 and coor[y, x, i] <= 3/5:
                        coor[y, x, i] = 5 * coor[y, x, i] - 2
                    else:
                        coor[y, x, i] = -5 * coor[y, x, i] + 4

        return coor

    def norm_coor_6(self, coor, obj_id):
        """
        normalize coordinates by object size in condition of spliting ply into 216 parts
        """
        coor[:, :, 0] = coor[:, :, 0] / abs(self.infos[obj_id]['min_x'])
        coor[:, :, 1] = coor[:, :, 1] / abs(self.infos[obj_id]['min_y'])
        coor[:, :, 2] = coor[:, :, 2] / abs(self.infos[obj_id]['min_z'])
        # for x in coor_x.shape[]
        for i in range(3):
            for x in range(64):
                for y in range(64):
                    if coor[y, x, i] <= -2/3:
                        coor[y, x, i] = -6 * coor[y, x, i] - 5
                    elif coor[y, x, i] > -2/3 and coor[y, x, i] <= -1/3:
                        coor[y, x, i] = 6 * coor[y, x, i] + 3
                    elif coor[y, x, i] > -1/3 and coor[y, x, i] <= 0:
                        coor[y, x, i] = -6 * coor[y, x, i] - 1
                    elif coor[y, x, i] > 0 and coor[y, x, i] <= 1/3:
                        coor[y, x, i] = 6 * coor[y, x, i] - 1
                    elif coor[y, x, i] > 1/3 and coor[y, x, i] <= 2/3:
                        coor[y, x, i] = -6 * coor[y, x, i] + 3
                    else:
                        coor[y, x, i] = 6 * coor[y, x, i] - 5

        return coor

    def norm_coor_7(self, coor, obj_id):
        """
        normalize coordinates by object size in condition of spliting ply into 343 parts
        """
        coor[:, :, 0] = coor[:, :, 0] / abs(self.infos[obj_id]['min_x'])
        coor[:, :, 1] = coor[:, :, 1] / abs(self.infos[obj_id]['min_y'])
        coor[:, :, 2] = coor[:, :, 2] / abs(self.infos[obj_id]['min_z'])
        # for x in coor_x.shape[]
        for i in range(3):
            for x in range(64):
                for y in range(64):
                    if coor[y, x, i] <= -5/7:
                        coor[y, x, i] = -7 * coor[y, x, i] - 6
                    elif coor[y, x, i] > -5/7 and coor[y, x, i] <= -3/7:
                        coor[y, x, i] = 7 * coor[y, x, i] + 4
                    elif coor[y, x, i] > -3/7 and coor[y, x, i] <= -1/7:
                        coor[y, x, i] = -7 * coor[y, x, i] - 2
                    elif coor[y, x, i] > -1/7 and coor[y, x, i] <= 1/7:
                        coor[y, x, i] = 7 * coor[y, x, i]
                    elif coor[y, x, i] > 1/7 and coor[y, x, i] <= 3/7:
                        coor[y, x, i] = -7 * coor[y, x, i] + 2
                    elif coor[y, x, i] > 3/7 and coor[y, x, i] <= 5/7:
                        coor[y, x, i] = 7 * coor[y, x, i] - 4
                    else:
                        coor[y, x, i] = -7 * coor[y, x, i] + 6

        return coor

    def norm_coor_8(self, coor, obj_id):
        """
        normalize coordinates by object size in condition of spliting ply into 512 parts
        """
        #coor_x, coor_y, coor_z = coor[..., 0], coor[..., 1], coor[..., 2]
        coor[:,:,0] = coor[:,:,0] / abs(self.infos[obj_id]['min_x'])
        coor[:,:,1] = coor[:,:,1] / abs(self.infos[obj_id]['min_y'])
        coor[:,:,2] = coor[:,:,2] / abs(self.infos[obj_id]['min_z'])
        # for x in coor_x.shape[]
        for i in range(3):
            for x in range(64):
                for y in range(64):
                    if coor[y, x, i] <= -0.75:
                        coor[y, x, i] = -8 * coor[y, x, i] - 7
                    elif coor[y, x, i] > -0.75 and coor[y, x, i] <= -0.5:
                        coor[y, x, i] = 8 * coor[y, x, i] + 5
                    elif coor[y, x, i] > -0.5 and coor[y, x, i] <= -0.25:
                        coor[y, x, i] = -8 * coor[y, x, i] - 3
                    elif coor[y, x, i] > -0.25 and coor[y, x, i] <= 0:
                        coor[y, x, i] = 8 * coor[y, x, i] + 1
                    elif coor[y, x, i] > 0 and coor[y, x, i] <= 0.25:
                        coor[y, x, i] = -8 * coor[y, x, i] + 1
                    elif coor[y, x, i] > 0.25 and coor[y, x, i] <= 0.5:
                        coor[y, x, i] = 8 * coor[y, x, i] - 3
                    elif coor[y, x, i] > 0.5 and coor[y, x, i] <= 0.75:
                        coor[y, x, i] = -8 * coor[y, x, i] + 5
                    else:
                        coor[y, x, i] = 8 * coor[y, x, i] - 7

        return coor

    def c_rel_delta(self, c_obj, c_box, wh_box):
        """
        compute relative bias between object center and bounding box center
        """
        c_delta = np.asarray(c_obj) - np.asarray(c_box)
        c_delta /= np.asarray(wh_box)
        return c_delta

    def d_scaled(self, depth, s_box, res):
        """
        compute scaled depth
        """
        r = float(res) / s_box
        return depth / r

    def __getitem__(self, idx):
        if self.split == 'train':
            obj = self.load_obj(idx)
            obj_id = ref.obj2idx(obj)
            data_type = self.load_type(idx)
            box = self.load_box(idx)
            pose = self.load_pose(idx)
            rgb = self.load_rgb(idx)
            msk_ori = self.load_msk(idx)
            if self.cfg.train.adjusted_mask:
                msk = self.load_new_msk(idx)
            else:
                msk = msk_ori

            if self.cfg.train.split_num == 2:
                msk_x1 = self.load_msk_x1(idx)
                msk_y1 = self.load_msk_y1(idx)
                msk_z1 = self.load_msk_z1(idx)
            elif self.cfg.train.split_num == 3:
                msk_x1 = self.load_msk_x1_N3(idx)
                msk_y1 = self.load_msk_y1_N3(idx)
                msk_z1 = self.load_msk_z1_N3(idx)
                msk_x2 = self.load_msk_x2_N3(idx)
                msk_y2 = self.load_msk_y2_N3(idx)
                msk_z2 = self.load_msk_z2_N3(idx)
            elif self.cfg.train.split_num == 4:          #4
                msk_x1 = self.load_msk_x1(idx)
                msk_y1 = self.load_msk_y1(idx)
                msk_z1 = self.load_msk_z1(idx)
                if self.cfg.train.no_bi == False:
                    msk_x2 = self.load_msk_x2(idx)
                    msk_y2 = self.load_msk_y2(idx)
                    msk_z2 = self.load_msk_z2(idx)
                else:
                    msk_x2 = self.load_msk_x2_nb(idx)
                    msk_y2 = self.load_msk_y2_nb(idx)
                    msk_z2 = self.load_msk_z2_nb(idx)
            elif self.cfg.train.split_num == 5:
                msk_x1 = self.load_msk_x1_N5(idx)
                msk_y1 = self.load_msk_y1_N5(idx)
                msk_z1 = self.load_msk_z1_N5(idx)
                msk_x2 = self.load_msk_x2_N5(idx)
                msk_y2 = self.load_msk_y2_N5(idx)
                msk_z2 = self.load_msk_z2_N5(idx)
                msk_x3 = self.load_msk_x3_N5(idx)
                msk_y3 = self.load_msk_y3_N5(idx)
                msk_z3 = self.load_msk_z3_N5(idx)
            elif self.cfg.train.split_num == 6:
                msk_x1 = self.load_msk_x1_N6(idx)
                msk_y1 = self.load_msk_y1_N6(idx)
                msk_z1 = self.load_msk_z1_N6(idx)
                msk_x2 = self.load_msk_x2_N6(idx)
                msk_y2 = self.load_msk_y2_N6(idx)
                msk_z2 = self.load_msk_z2_N6(idx)
                msk_x3 = self.load_msk_x3_N6(idx)
                msk_y3 = self.load_msk_y3_N6(idx)
                msk_z3 = self.load_msk_z3_N6(idx)
            elif self.cfg.train.split_num == 7:
                msk_x1 = self.load_msk_x1_N7(idx)
                msk_y1 = self.load_msk_y1_N7(idx)
                msk_z1 = self.load_msk_z1_N7(idx)
                msk_x2 = self.load_msk_x2_N7(idx)
                msk_y2 = self.load_msk_y2_N7(idx)
                msk_z2 = self.load_msk_z2_N7(idx)
                msk_x3 = self.load_msk_x3_N7(idx)
                msk_y3 = self.load_msk_y3_N7(idx)
                msk_z3 = self.load_msk_z3_N7(idx)
            elif self.cfg.train.split_num == 8:          # 8
                msk_x1 = self.load_msk_x1(idx)
                msk_y1 = self.load_msk_y1(idx)
                msk_z1 = self.load_msk_z1(idx)
                msk_x2 = self.load_msk_x2(idx)
                msk_y2 = self.load_msk_y2(idx)
                msk_z2 = self.load_msk_z2(idx)
                msk_x3 = self.load_msk_x3(idx)
                msk_y3 = self.load_msk_y3(idx)
                msk_z3 = self.load_msk_z3(idx)

            coor = self.load_coor(idx)
            if self.split == 'train':
                if (self.annot[idx]['data_type']=='imgn') or (random.random()<self.cfg.augment.change_bg_ratio):            # imgn all changed, real images half half
                    try:
                        rgb = self.change_bg(rgb, msk_ori)
                    except:
                        print(self.annot[idx]['coor_pth'])
            if (self.split == 'train') and self.cfg.dataiter.dzi and not self.cfg.train.err_res:
                c, s = self.xywh_to_cs_dzi(box, self.cfg.augment.pad_ratio, s_max=max(ref.im_w, ref.im_h))
            else:
                c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(ref.im_w, ref.im_h))
            if self.cfg.dataiter.denoise_coor:
                coor = self.denoise_coor(coor)

            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            msk, *_ = zoom_in(msk, c, s, self.cfg.dataiter.out_res, channel=1)
            if self.cfg.train.split_num > 1:
                msk_x1, *_ = zoom_in(msk_x1, c, s, self.cfg.dataiter.out_res, channel=1)
                msk_y1, *_ = zoom_in(msk_y1, c, s, self.cfg.dataiter.out_res, channel=1)
                msk_z1, *_ = zoom_in(msk_z1, c, s, self.cfg.dataiter.out_res, channel=1)
            if self.cfg.train.split_num > 2:
                msk_x2, *_ = zoom_in(msk_x2, c, s, self.cfg.dataiter.out_res, channel=1)
                msk_y2, *_ = zoom_in(msk_y2, c, s, self.cfg.dataiter.out_res, channel=1)
                msk_z2, *_ = zoom_in(msk_z2, c, s, self.cfg.dataiter.out_res, channel=1)
            if self.cfg.train.split_num > 4:
                msk_x3, *_ = zoom_in(msk_x3, c, s, self.cfg.dataiter.out_res, channel=1)
                msk_y3, *_ = zoom_in(msk_y3, c, s, self.cfg.dataiter.out_res, channel=1)
                msk_z3, *_ = zoom_in(msk_z3, c, s, self.cfg.dataiter.out_res, channel=1)
            coor, *_ = zoom_in(coor, c, s, self.cfg.dataiter.out_res, interpolate=cv2.INTER_NEAREST)    ###
            #print('coor = ', coor)
            c = np.array([c_w_, c_h_])
            s = s_
            if self.cfg.train.split_num == 1:
                coor = self.norm_coor(coor, obj_id).transpose(2, 0, 1)
            elif self.cfg.train.split_num == 2:
                coor = self.norm_coor_2(coor, obj_id).transpose(2, 0, 1)
            elif self.cfg.train.split_num == 3:
                coor = self.norm_coor_3(coor, obj_id).transpose(2, 0, 1)
            elif self.cfg.train.split_num == 4:
                coor = self.norm_coor_4(coor, obj_id).transpose(2, 0, 1)
            elif self.cfg.train.split_num == 5:
                coor = self.norm_coor_5(coor, obj_id).transpose(2, 0, 1)
            elif self.cfg.train.split_num == 6:
                coor = self.norm_coor_6(coor, obj_id).transpose(2, 0, 1)
            elif self.cfg.train.split_num == 7:
                coor = self.norm_coor_7(coor, obj_id).transpose(2, 0, 1)
            elif self.cfg.train.split_num == 8:
                coor = self.norm_coor_8(coor, obj_id).transpose(2, 0, 1)
            try:
                inp = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            except:
                print(self.annot[idx]['rgb_pth'])
            if self.cfg.train.split_num == 1:
                out = np.concatenate([coor, msk[None, :, :]], axis=0)
                loss_msk = np.stack([msk, msk, msk, np.ones_like(msk)], axis=0)
            elif self.cfg.train.split_num == 2:
                out = np.concatenate([coor, msk[None, :, :], msk_x1[None, :, :], msk_y1[None, :, :], msk_z1[None, :, :]], axis=0)
                msk_for_split = self.cfg.train.mask_xyz_cof * np.ones_like(msk)
                loss_msk = np.stack([msk, msk, msk, msk_for_split, msk_for_split, msk_for_split, msk_for_split], axis=0)
            elif self.cfg.train.split_num > 2 and self.cfg.train.split_num <= 4:
                out = np.concatenate([coor, msk[None, :, :], msk_x1[None, :, :], msk_y1[None, :, :], msk_z1[None, :, :], \
                                                             msk_x2[None, :, :], msk_y2[None, :, :], msk_z2[None, :, :]], axis=0)
                msk_for_split = self.cfg.train.mask_xyz_cof * np.ones_like(msk)
                loss_msk = np.stack([msk, msk, msk, msk_for_split, msk_for_split, msk_for_split, msk_for_split, msk_for_split, msk_for_split, msk_for_split], axis=0)
                '''
                out = np.concatenate([coor, msk[None, :, :], msk_x1[None, :, :], msk_y1[None, :, :], msk_z1[None, :, :], \
                                                             msk_x2[None, :, :], msk_y2[None, :, :], msk_z2[None, :, :]], axis=0)
                msk_for_split = self.cfg.train.mask_xyz_cof * np.ones_like(msk_ori)
                loss_msk = np.stack([msk_ori, msk_ori, msk_ori, msk_for_split, msk_for_split, msk_for_split, msk_for_split, msk_for_split, msk_for_split, msk_for_split], axis=0)
                '''
            elif self.cfg.train.split_num > 4 and self.cfg.train.split_num <= 8:
                out = np.concatenate([coor, msk[None, :, :], msk_x1[None, :, :], msk_y1[None, :, :], msk_z1[None, :, :], \
                                                             msk_x2[None, :, :], msk_y2[None, :, :], msk_z2[None, :, :], \
                                                             msk_x3[None, :, :], msk_y3[None, :, :], msk_z3[None, :, :]], axis=0)
                msk_for_split = self.cfg.train.mask_xyz_cof * np.ones_like(msk)
                loss_msk = np.stack([msk, msk, msk, msk_for_split, msk_for_split, msk_for_split, msk_for_split, msk_for_split, \
                                                        msk_for_split, msk_for_split, msk_for_split, msk_for_split, msk_for_split], axis=0)

            trans = pose[:, 3]    # in this, trans is the array which size is (1, 3)
            c_obj, _ = prj_vtx_cam(trans, self.cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)
            c_obj_pts = np.array([32. + c_delta[0] * 64., 32. + c_delta[1] * 64.])
            if self.ger_fig == True:
                plt.scatter(c_obj_pts[0], c_obj_pts[1], s=5, c='purple')
                plt.savefig('/home/zyz/文档/c_fig_lm.png')
            return obj, obj_id, inp, out, loss_msk, trans_local, pose, c, s, np.asarray(box)

        if self.split == 'test':
            obj = self.load_obj(idx)
            obj_id = ref.obj2idx(obj)
            box = self.load_box(idx)
            pose = self.load_pose(idx)
            rgb = self.load_rgb(idx)
            c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(ref.im_w, ref.im_h))
            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            c = np.array([c_w_, c_h_])
            s = s_
            inp = rgb
            trans = pose[:, 3]
            c_obj, _ = prj_vtx_cam(trans, self.cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)
            depth_test_pth = self.annot[idx]['depth_pth']
            return obj, obj_id, inp, pose, c, s, np.asarray(box), trans_local, depth_test_pth

    def __len__(self):
        return self.num

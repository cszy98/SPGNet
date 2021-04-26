from __future__ import division
import torch
import torchvision.transforms as transforms
from .base_dataset import *
import cv2
import numpy as np
import os
import util.io as io

class PoseTransferParsingDataset(BaseDataset):
    def name(self):
        return 'PoseTransferParsingDataset'

    def initialize(self, opt, split):
        self.opt = opt
        self.data_root = opt.data_root
        self.split = split
        #############################
        # set path / load label
        #############################
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        self.img_dir = os.path.join(opt.data_root, opt.img_dir)
        self.seg_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.pose_label = io.load_data(os.path.join(opt.data_root, opt.fn_pose))

        self.seg_cihp_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.seg_cihp_pred_dir = os.path.join(opt.data_root, opt.seg_pred_dir)

        #############################
        # create index list
        #############################
        self.id_list = data_split[split] if split in data_split.keys() else data_split['test'][:2000]
        self._len = len(self.id_list)
        #############################
        # other
        #############################
        # here set debug option
        if opt.debug:
            self.id_list = self.id_list[0:32]
        self.tensor_normalize_std = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.to_pil_image = transforms.ToPILImage()
        self.pil_to_tensor = transforms.ToTensor()
        self.color_jitter = transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.2)

    def set_len(self, n):
        self._len = n

    def __len__(self):
        if hasattr(self, '_len') and self._len > 0:
            return self._len
        else:
            return len(self.id_list)

    def to_tensor(self, np_data):
        return torch.Tensor(np_data.transpose((2, 0, 1)))

    def read_image(self, sid):
        fn = os.path.join(self.img_dir, sid + '.jpg')
        # print(fn)
        # print(os.path.exists(fn))
        img = cv2.imread(fn).astype(np.float32) / 255.
        img = img[..., [2, 1, 0]]
        return img

    def read_seg_pred_cihp(self, sid1, sid2):
        fn = os.path.join(self.seg_cihp_pred_dir, sid1 + '___' + sid2 + '.png')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[...,np.newaxis]
        return seg

    def read_seg_cihp(self, sid):
        fn = os.path.join(self.seg_cihp_dir, sid + '.png')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[..., np.newaxis]
        return seg

    def __getitem__(self, index):
        sid1, sid2 = self.id_list[index]
        ######################
        # load data
        ######################
        img_1 = self.read_image(sid1)
        img_2 = self.read_image(sid2)

        seg_cihp_label_1 = self.read_seg_cihp(sid1)
        seg_cihp_label_2 = self.read_seg_cihp(sid2) if self.split=='train' else self.read_seg_pred_cihp(sid1, sid2)
        joint_c_1 = np.array(self.pose_label[sid1])
        joint_c_2 = np.array(self.pose_label[sid2])
        h, w = self.opt.image_size
        ######################
        # pack output data
        ######################
        joint_1 = kp_to_map(img_sz=(w, h), kps=joint_c_1, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        joint_2 = kp_to_map(img_sz=(w, h), kps=joint_c_2, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        seg_cihp_1 = seg_label_to_map(seg_cihp_label_1, nc=20)
        seg_cihp_2 = seg_label_to_map(seg_cihp_label_2, nc=20)

        data = {
            'img_1': self.tensor_normalize_std(self.to_tensor(img_1)),
            'img_2': self.tensor_normalize_std(self.to_tensor(img_2)),
            'joint_1': self.to_tensor(joint_1),
            'joint_2': self.to_tensor(joint_2),
            'seg_cihp_1': self.to_tensor(seg_cihp_1),
            'seg_cihp_2': self.to_tensor(seg_cihp_2),
            'id_1': sid1,
            'id_2': sid2
        }
        return data
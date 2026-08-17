# -*- coding: utf-8 -*-
# @Description: Data preprocessing and organization for Tanks and Temples dataset.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import os
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from datasets.data_io import *  # NOQA


class MVSDataset(Dataset):
    def __init__(self, datapath, list_file, split, n_views, ndepths=192, interval_scale=1.06, inverse_depth=False, **kwargs):
        super(MVSDataset, self).__init__()

        self.datapath = datapath
        self.list_file = list_file
        self.split = split
        self.n_views = n_views
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth

        self.cam_mode = kwargs.get("cam_mode", "origin")    # origin / short_range
        if self.cam_mode == 'short_range':
            assert self.split == "intermediate"
        self.img_mode = kwargs.get("img_mode", "resize")    # resize / crop

        self.total_depths = ndepths
        self.depth_interval_table = {
            # intermediate
            'Family': 2.5e-3, 'Francis': 1e-2, 'Horse': 1.5e-3, 'Lighthouse': 1.5e-2, 'M60': 5e-3, 'Panther': 5e-3, 'Playground': 7e-3, 'Train': 5e-3,
            # advanced
            'Auditorium': 3e-2, 'Ballroom': 2e-2, 'Courtroom': 2e-2, 'Museum': 2e-2, 'Palace': 1e-2, 'Temple': 1e-2
        }
        self.img_wh = kwargs.get("img_wh", (-1, 1024))

        self.metas = self.build_list()

    def build_list(self):
        metas = []
        scans = self.list_file

        interval_scale_dict = {}
        # scans
        with open(os.path.join(self.list_file)) as f:
            scans = [line.rstrip() for line in f.readlines()]
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, self.split, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.n_views - 1:
                            print("{}< src num_views:{}".format(len(src_views), self.n_views))
                            src_views += [src_views[0]] * (self.n_views - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        print("dataset", "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))

        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])
        depth_interval = float(lines[11].split()[1])
        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.total_depths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_max, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def scale_tnt_input(self, intrinsics, img):
        if self.img_mode == "crop":
            intrinsics[1, 2] = intrinsics[1, 2] - 28  # 1080 -> 1024
            img = img[28:1080 - 28, :, :]
        elif self.img_mode == "resize":
            height, width = img.shape[:2]

            max_w, max_h = self.img_wh[0], self.img_wh[1]
            if max_w == -1:
                max_w = width

            img = cv2.resize(img, (max_w, max_h))

            scale_w = 1.0 * max_w / width
            intrinsics[0, :] *= scale_w
            scale_h = 1.0 * max_h / height
            intrinsics[1, :] *= scale_h

        return intrinsics, img

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        view_ids = [ref_view] + src_views[:self.n_views - 1]

        imgs = []
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, self.split, scan, f'images/{vid:08d}.jpg')
            if self.cam_mode == 'short_range':
                # can only use for Intermediate
                proj_mat_filename = os.path.join(self.datapath, self.split, scan, f'cams_{scan.lower()}/{vid:08d}_cam.txt')
            elif self.cam_mode == 'origin':
                proj_mat_filename = os.path.join(self.datapath, self.split, scan, f'cams/{vid:08d}_cam.txt')

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min_, depth_max_, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=self.interval_scale[scene_name])
            intrinsics, img = self.scale_tnt_input(intrinsics, img)
            imgs.append(img.transpose(2, 0, 1))

            proj_mat_0 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_1 = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_2 = np.zeros(shape=(2, 4, 4), dtype=np.float32)

            intrinsics[:2, :] *= 0.25
            proj_mat_0[0, :4, :4] = extrinsics.copy()
            proj_mat_0[1, :3, :3] = intrinsics.copy()
            int_mat_0 = intrinsics.copy()

            intrinsics[:2, :] *= 2
            proj_mat_1[0, :4, :4] = extrinsics.copy()
            proj_mat_1[1, :3, :3] = intrinsics.copy()
            int_mat_1 = intrinsics.copy()

            intrinsics[:2, :] *= 2
            proj_mat_2[0, :4, :4] = extrinsics.copy()
            proj_mat_2[1, :3, :3] = intrinsics.copy()
            int_mat_2 = intrinsics.copy()

            proj_matrices_0.append(proj_mat_0)
            proj_matrices_1.append(proj_mat_1)
            proj_matrices_2.append(proj_mat_2)

            # reference view
            if i == 0:
                depth_min = depth_min_
                if self.cam_mode == 'short_range':
                    depth_max = depth_min + self.total_depths * self.depth_interval_table[scan]
                elif self.cam_mode == 'origin':
                    depth_max = depth_max_
                if self.inverse_depth:
                    depth_end = depth_interval * self.total_depths + depth_min
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_end, self.total_depths, endpoint=False)
                    depth_values = (1.0 / depth_values).astype(np.float32)
                    init_depth_hypotheses = depth_values
                else:
                    depth_values = np.arange(depth_min, depth_interval * (self.total_depths - 0.5) + depth_min,
                                             depth_interval,
                                             dtype=np.float32)
                    init_depth_hypotheses = np.arange(depth_min, depth_interval * (self.total_depths - 0.5) + depth_min, depth_interval,
                                                      dtype=np.float32)
        imgs = np.stack(imgs)
        proj = {}
        proj['stage1'] = np.stack(proj_matrices_0)
        proj['stage2'] = np.stack(proj_matrices_1)
        proj['stage3'] = np.stack(proj_matrices_2)

        intrinsics_matrices = {
            "stage1": int_mat_0,
            "stage2": int_mat_1,
            "stage3": int_mat_2
        }

        sample = {
            "imgs": imgs,
            "proj_matrices": proj,
            "intrinsics_matrices": intrinsics_matrices,
            "depth_values": depth_values,
            "init_depth_hypotheses": init_depth_hypotheses,
            "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
        }

        return sample

# -*- coding: utf-8 -*-
# @Description: Point cloud fusion strategy for DTU dataset: Basic PCD.
#     Refer to: https://github.com/xy-guo/MVSNet_pytorch/blob/master/eval.py
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import argparse, os, sys, cv2, re, logging, time  # NOQA
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image

from multiprocessing import Pool
from functools import partial
import signal
from tank_test_config import tank_cfg


parser = argparse.ArgumentParser(description='filter, and fuse')

parser.add_argument('--testpath', default='/home/lida/lida/2Ddataset/tankandtemples/', help='testing data dir for some scenes')
parser.add_argument('--testlist', default='datasets/lists/tnt/intermediate.txt', choices=['intermediate', 'advanced'], help='testing scene list')  # test_copy
parser.add_argument('--split', type=str, default='intermediate', choices=['intermediate', 'advanced'])
parser.add_argument('--method', type=str, default='casmvs')
parser.add_argument('--outdir', default='./outputs/tnt/jiayou_test', help='output dir')
parser.add_argument('--img_mode', type=str, default='crop', choices=['resize', 'crop'])

args = parser.parse_args()


def read_pair_file(filename, dataset="tnt"):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for _ in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            if dataset!="eth3d":
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            else:
                src_views = []
                src_views_score = [float(x) for x in f.readline().rstrip().split()]
                src_views_1 = src_views_score[1::2]
                src_views_1 = [int(x) for x in src_views_1]
                score = src_views_score[2::2]
                for i in range(len(src_views_1)):
                    if score[i]>0.1 and src_views_1[i]!=ref_view:
                        src_views.append(src_views_1[i])
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    depth_min = float(lines[11].split()[1])
    depth_max = float(lines[11].split()[0])
    # DTU dataset
    # TODO: seems not needed to hard-code
    if depth_max>425:
        depth_max = 935
        depth_min = 425
    return intrinsics, extrinsics, depth_max, depth_min


def scale_input(intrinsics, img):
    if args.img_mode == "crop":
        intrinsics[1, 2] = intrinsics[1, 2] - 28  # 1080 -> 1024
        img = img[28:1080 - 28, :, :]
    elif args.img_mode == "resize":
        height, width = img.shape[:2]
        img = cv2.resize(img, (width, 1024))
        scale_h = 1.0 * 1024 / height
        intrinsics[1, :] *= scale_h

    return intrinsics, img


def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
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


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_mask(filename, mask):
    assert mask.dtype == np.bool_
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# project the reference point cloud into the source view, then project back
def reproject_with_depth(
    depth_ref,
    intrinsics_ref,
    extrinsics_ref,
    depth_src,
    intrinsics_src,
    extrinsics_src
):
    """project the reference point cloud into the source view, then project back"""
    width, height = depth_ref.shape[1], depth_ref.shape[0]

    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref),
        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1])
    )

    # coordniate transformation
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src,
                                  interpolation=cv2.INTER_LINEAR)

    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src),
        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1])
    )

    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    depth_reproj = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected = np.where(K_xyz_reprojected == 0, 1e-5, K_xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    xy_reprojected = np.clip(xy_reprojected, -1e8, 1e8)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reproj, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(
    depth_ref,
    intrinsics_ref,
    extrinsics_ref,
    depth_src,
    intrinsics_src,
    extrinsics_src,
    ref_depth_max,
    ref_depth_min,
    geo_pixel_thres=1.0,
    geo_depth_thres=0.01,
    ):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reproj, x2d_reproj, y2d_reproj, x2d_src, y2d_src = reproject_with_depth(
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src
    )

    dist = np.sqrt((x2d_reproj - x_ref) ** 2 + (y2d_reproj - y_ref) ** 2)

    depth_diff = np.abs(depth_reproj - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, 
                            relative_depth_diff < geo_depth_thres)
    mask2 = np.logical_and(depth_ref > ref_depth_min, depth_ref < ref_depth_max)
    mask = np.logical_and(mask, mask2)
    depth_reproj[~mask] = 0
    return mask, depth_reproj, x2d_src, y2d_src


def filter_depth(
    pair_folder,
    out_folder,
    plyfilename,
    geo_mask_thres=3,
    geo_pixel_thres=1.0,
    geo_depth_thres=0.01,
    photo_thres=[0.0, 0.1, 0.3, 0.5],
    method='casmvs',
    dataset='tnt',
):
    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    pair_data = read_pair_file(pair_file, dataset)

    vertexs = []
    vertex_colors = []

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        ref_intrinsics, ref_extrinsics, depth_max, depth_min = read_camera_parameters(
            os.path.join(pair_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(pair_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        ref_depth_est = read_pfm(
            os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        
        """photometric filtering"""
        if method == 'casmvs':
            confidence3 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
            confidence2 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage3.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage2.pfm'.format(ref_view)))[0]
            confidence0 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage1.pfm'.format(ref_view)))[0]

            photo_mask0 = confidence0 > photo_thres[0]
            photo_mask1 = confidence1 > photo_thres[1]
            photo_mask2 = confidence2 > photo_thres[2]
            photo_mask3 = confidence3 > photo_thres[3]
            photo_mask = photo_mask0 & photo_mask1 & photo_mask2 & photo_mask3
        else:
            confidence0 = read_pfm(
                os.path.join(out_folder, 'conf0/{:0>8}.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(
                os.path.join(out_folder, 'conf1/{:0>8}.pfm'.format(ref_view)))[0]

            photo_mask0 = confidence0 > photo_thres[0]
            photo_mask1 = confidence1 > photo_thres[1]
            photo_mask = photo_mask0 & photo_mask1
        flag_img = ref_img
        ref_intrinsics, _ = scale_input(ref_intrinsics, flag_img)
        """geometric filtering"""
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        geo_mask_sum = 0

        for i, src_view in enumerate(src_views):
            src_intrinsics, src_extrinsics, _, _ = read_camera_parameters(
                os.path.join(pair_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))

            src_depth_est = read_pfm(
                os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            src_intrinsics, _ = scale_input(src_intrinsics, flag_img)
            geo_mask, depth_reproj, x2d_src, y2d_src = check_geometric_consistency(
                ref_depth_est,
                ref_intrinsics,
                ref_extrinsics,
                src_depth_est,
                src_intrinsics, 
                src_extrinsics,
                depth_max,
                depth_min,
                geo_pixel_thres,
                geo_depth_thres
            )
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reproj)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= geo_mask_thres

        final_mask = np.logical_and(photo_mask, geo_mask)
        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_photo.jgp".format(ref_view)),
            photo_mask
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_geo.jgp".format(ref_view)),
            geo_mask
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_final.jgp".format(ref_view)),
            final_mask
        )

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(
            out_folder,
            ref_view,
            photo_mask.mean(),
            geo_mask.mean(),
            final_mask.mean())
        )

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        
        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors],
                             dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def check_geometric_consistency_dynamic(
    depth_ref,
    intrinsics_ref,
    extrinsics_ref,
    depth_src,
    intrinsics_src,
    extrinsics_src,
    dh_pixel_dist_num
):
    """dynamic filtering for tanks & temples"""
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reproj, x2d_reproj, y2d_reproj, x2d_src, y2d_src = reproject_with_depth(
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src
    )
    dist = np.sqrt((x2d_reproj - x_ref) ** 2 + (y2d_reproj - y_ref) ** 2)
    depth_diff = np.abs(depth_reproj - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks = []
    for i in range(dh_pixel_dist_num[0], 11):
        mask = np.logical_and(dist < i / dh_pixel_dist_num[1],
                              relative_depth_diff < i / dh_pixel_dist_num[2])
        masks.append(mask)
    depth_reproj[~mask] = 0

    return masks, mask, depth_reproj, x2d_src, y2d_src

def filter_depth_dynamic(
    scan,
    pair_folder,
    out_folder,
    plyfilename,
    photo_thres=[0.0, 0.1, 0.3, 0.5],
    method='casmvs',
    dataset='tank'
):
    """dynamic filtering for tanks & temples"""

    dh_view_num_all = {
        'Family':2, 'Francis':9, 'Horse':2,
        'Lighthouse':6, 'M60':4, 'Panther':3,
        'Playground':6, 'Train':3,
        'Auditorium':2, 'Ballroom':2, 'Courtroom':2,
        'Museum':2, 'Palace':2, 'Temple':1
    }
    dist_all = {
        'Family':12, 'Francis':8, 'Horse':4,
        'Lighthouse':8, 'M60':8, 'Panther':4,
        'Playground':8, 'Train':4,
        'Auditorium':4, 'Ballroom':4, 'Courtroom':4,
        'Museum':4, 'Palace':4, 'Temple':4
    }
    rel_diff_all = {
        'Family':1600, 'Francis':1600, 'Horse':1300,
        'Lighthouse':1600, 'M60':1600, 'Panther':1300,
        'Playground':1600, 'Train':1600,
        'Auditorium':1300, 'Ballroom':1300, 'Courtroom':1300,
        'Museum':1300, 'Palace':1300, 'Temple':1500
    }

    photo_thres_all = {
        "Family": [0.8, 0.8, 0.95],
        "Francis": [0.3, 0.6, 0.6],
        "Horse": [0.15, 0.4, 0.8],
        "Lighthouse": [0.3, 0.8, 0.9],
        "M60": [0.7, 0.8, 0.95],
        "Panther": [0.3, 0.3, 0.95],
        "Playground": [0.3, 0.8, 0.9],
        "Train": [0.3, 0.6, 0.95],
        "Auditorium": [0., 0., 0.],
        "Ballroom": [0.3, 0.3, 0.5],
        "Courtroom": [0., 0.2, 0.2],
        "Museum": [0.3, 0.3, 0.7],
        "Palace": [0.3, 0.3, 0.4],
        "Temple": [0.3, 0.5, 0.5],
    }
    
    dh_view_num = dh_view_num_all[scan] 
    dh_dist = dist_all[scan]
    dh_rel_diff = rel_diff_all[scan]
    dh_pixel_dist_num = [dh_view_num, dh_dist, dh_rel_diff] 

    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    ct2 = -1
    for ref_view, src_views in pair_data:
        ct2 += 1
        # load the camera parameters
        ref_intrinsics, ref_extrinsics, ref_depth_max, ref_depth_min = read_camera_parameters(
            os.path.join(pair_folder, 'cams/{:0>8}_cam.txt'.format(ref_view))
        )
        # load the reference image
        ref_img = read_img(os.path.join(pair_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder,
                                              'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        flag_img = ref_img
        ref_intrinsics, _ = scale_input(ref_intrinsics, flag_img)
        """photometric filtering"""
        if method == 'casmvs':
            confidence3 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
            confidence2 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage3.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage2.pfm'.format(ref_view)))[0]
            confidence0 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage1.pfm'.format(ref_view)))[0]

            photo_mask0 = confidence0 > photo_thres[0]
            photo_mask1 = confidence1 > photo_thres[1]
            photo_mask2 = confidence2 > photo_thres[2]
            photo_mask3 = confidence3 > photo_thres[3]
            photo_mask = photo_mask0 & photo_mask1 & photo_mask2 & photo_mask3
        else:
            confidence0 = read_pfm(
                os.path.join(out_folder, 'conf0/{:0>8}.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(
                os.path.join(out_folder, 'conf1/{:0>8}.pfm'.format(ref_view)))[0]

            photo_mask0 = confidence0 > photo_thres[0]
            photo_mask1 = confidence1 > photo_thres[2] # use the last threshold for the final refinement
            photo_mask = photo_mask0 & photo_mask1

        """geometric filtering following D2HC-RMVSNet"""
        all_srcview_depth_ests = []
        geo_mask_sum = 0
        geo_mask_sums = []
        ct = 0
        for src_view in src_views:
            ct = ct + 1
            # camera parameters of the source view
            src_intrinsics, src_extrinsics, _, _  = read_camera_parameters(
                os.path.join(pair_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(
                os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            src_intrinsics, _ = scale_input(src_intrinsics, flag_img)
            masks, geo_mask, depth_reproj, _, _ = check_geometric_consistency_dynamic(
                ref_depth_est,
                ref_intrinsics,
                ref_extrinsics,
                src_depth_est,
                src_intrinsics,
                src_extrinsics,
                dh_pixel_dist_num
            )

            if (ct == 1):
                for i in range(dh_view_num, 11):
                    geo_mask_sums.append(masks[i-dh_view_num].astype(np.int32))
            else:
                for i in range(dh_view_num, 11):
                    geo_mask_sums[i - dh_view_num] += masks[i-dh_view_num].astype(np.int32)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reproj)

        geo_mask = geo_mask_sum >= 10
        for i in range(dh_view_num, 11):
            geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - dh_view_num] >= i)

        depth_est_averaged = ((sum(all_srcview_depth_ests) + ref_depth_est) /
                              (geo_mask_sum + 1))
        maskdepth = np.logical_and(depth_est_averaged >= ref_depth_min,
                                   depth_est_averaged <= ref_depth_max)

        final_mask = np.logical_and(photo_mask, geo_mask)
        final_mask = np.logical_and(final_mask, maskdepth)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_photo.jgp".format(ref_view)), 
            photo_mask
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_geo.jgp".format(ref_view)),
            geo_mask
        )
        save_mask(
            os.path.join(out_folder, "mask/{:0>8}_final.jgp".format(ref_view)), 
            final_mask
        )

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(
            out_folder,
            ref_view,
            photo_mask.mean(),
            geo_mask.mean(),
            final_mask.mean()
        ))

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[:, :, :][valid_points]
        color = ref_img[28:1080 - 28, :, :][valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors],
                             dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    with open(args.testlist) as f:
        content = f.readlines()
        testlist = [line.rstrip() for line in content]
    
    photo_thres_all = {
        "Family": [0.3, 0.3, 0.3, 0.6],
        "Francis": [0.15, 0.3, 0.6, 0.6],
        "Horse": [0.1, 0.15, 0.4, 0.8],
        "Lighthouse": [0.15, 0.3, 0.8, 0.9],
        "M60": [0.35, 0.7, 0.8, 0.95],
        "Panther": [0.15, 0.3, 0.3, 0.95],
        "Playground": [0.15, 0.3, 0.8, 0.9],
        "Train": [0.15, 0.3, 0.6, 0.95],
        "Auditorium": [0., 0., 0., 0.],
        "Ballroom": [0.15, 0.3, 0.3, 0.5],
        "Courtroom": [0., 0., 0.2, 0.2],
        "Museum": [0.15, 0.3, 0.3, 0.7],
        "Palace": [0.1, 0.3, 0.3, 0.4],
        "Temple": [0.1, 0.3, 0.5, 0.5],
    }
    for scan in testlist:
        pair_folder = os.path.join(args.testpath, args.split, scan)
        out_folder = os.path.join(args.outdir, scan)
        plypath = args.outdir + '/pc'
        if not os.path.exists(plypath):
            os.makedirs(plypath)
        # scan = scan.split('/')[1]
        plyfilename = os.path.join(args.outdir, 'pc/{}.ply'.format(scan))
        filter_depth_dynamic(
            scan,
            pair_folder,
            out_folder,
            plyfilename,
            photo_thres_all[scan],
            args.method,
        )
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
parser.add_argument('--testlist', default='lists/tnt/intermediate.txt', help='testing scene list')  # test_copy
parser.add_argument('--split', type=str, default='intermediate', choices=['intermediate', 'advanced'])

parser.add_argument('--outdir', default='./outputs/tnt/jiayou_test', help='output dir')
parser.add_argument('--logdir', default='./outputs/tnt/jiayou_test', help='the directory to save checkpoints/logs')
parser.add_argument('--nolog', action='store_true', help='do not logging into .log file')
parser.add_argument('--plydir', default='./outputs/tnt/jiayou_test/dypcd_fusion_plys_tnt/', help='output dir')

parser.add_argument('--num_worker', type=int, default=1, help='depth_filer worker')
parser.add_argument('--single_processor', default=True, action='store_true')

parser.add_argument('--conf', type=float, nargs='+', default=[0.0, 0.0, 0.15, 0.0], help='prob confidence, for pcd and dypcd')  # [0.1, 0.1, 0.15, 0.5]
parser.add_argument('--levels', type=int, default=4, help='num of stages')
parser.add_argument('--display', default=False, action='store_true', help='display depth images and masks')
parser.add_argument('--img_mode', type=str, default='resize', choices=['resize', 'nresize'])  # 设置重新计算相机参数
parser.add_argument('--dist_base', type=float, default=1 / 4)
parser.add_argument('--rel_diff_base', type=float, default=1 / 1300)

args = parser.parse_args()


if args.split == 'intermediate':
    s_all = {'Family':2, 'Francis':7, 'Horse':2, 'Lighthouse':3, 'M60':3, 'Panther':2, 'Playground':3, 'Train':3}
    # conf_all = {'Family':0.35, 'Francis':0.4, 'Horse':0.2, 'Lighthouse':0.6, 'M60':0.4, 'Panther':0.3, 'Playground':0.4, 'Train':0.4}
    dist_all = {'Family':1/8, 'Francis':1/5, 'Horse':1/2, 'Lighthouse':1/4, 'M60':1/5, 'Panther':1/4, 'Playground':1/4, 'Train':1/4}
    rel_diff_all = {'Family':1/1600, 'Francis':1/1600, 'Horse':1/1300, 'Lighthouse':1/1000, 'M60':1/1600, 'Panther':1/1600, 'Playground':1/1600, 'Train':1/1600}
elif args.split == 'advanced':
    s_all = {'Auditorium':1, 'Ballroom':2, 'Courtroom':1, 'Museum':1, 'Palace':2, 'Temple':1}
    # conf_all = {'Auditorium':0.1, 'Ballroom':0.05, 'Courtroom':0.2, 'Museum':0.25, 'Palace':0.15, 'Temple':0.15}
    dist_all = {'Auditorium':1/2, 'Ballroom':1/4, 'Courtroom':1/5, 'Museum':1/5, 'Palace':1/4, 'Temple':1/4}
    rel_diff_all = {'Auditorium':1/1300, 'Ballroom':1/1300, 'Courtroom':1/1500, 'Museum':1/1500, 'Palace':1/1300, 'Temple':1/1500}


# if args.split == 'intermediate':
#     s_all = {'Family':2, 'Francis':7, 'Horse':2, 'Lighthouse':3, 'M60':3, 'Panther':2, 'Playground':3, 'Train':3}
#     # conf_all = {'Family':0.35, 'Francis':0.4, 'Horse':0.2, 'Lighthouse':0.6, 'M60':0.4, 'Panther':0.3, 'Playground':0.4, 'Train':0.4}
#     dist_all = {'Family':1/10, 'Francis':1/8, 'Horse':1/4, 'Lighthouse':1/4, 'M60':1/8, 'Panther':1/4, 'Playground':1/4, 'Train':1/4}
#     rel_diff_all = {'Family':1/1600, 'Francis':1/1600, 'Horse':1/1300, 'Lighthouse':1/1000, 'M60':1/1600, 'Panther':1/1500, 'Playground':1/1600, 'Train':1/1600}
# elif args.split == 'advanced':
#     s_all = {'Auditorium':1, 'Ballroom':2, 'Courtroom':1, 'Museum':1, 'Palace':1, 'Temple':1}
#     # conf_all = {'Auditorium':0.1, 'Ballroom':0.05, 'Courtroom':0.2, 'Museum':0.25, 'Palace':0.15, 'Temple':0.15}
#     dist_all = {'Auditorium':1/2, 'Ballroom':1/4, 'Courtroom':1/4, 'Museum':1/4, 'Palace':1/4, 'Temple':1/4}
#     rel_diff_all = {'Auditorium':1/1000, 'Ballroom':1/1300, 'Courtroom':1/1500, 'Museum':1/1500, 'Palace':1/1500, 'Temple':1/1500}


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


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool_
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read ref_depth_max, ref_depth_min
def read_depth_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix

    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[-1])
    return depth_max, depth_min


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


def read_mask(filename):
    return read_img(filename) > 0.5


def calculate_percentile(confidence, percent):
    percentile_80 = np.percentile(confidence, percent)
    return percentile_80


def scale_input(intrinsics, img):
    height, width = img.shape[:2]
    img = cv2.resize(img, (width, 1024))
    scale_h = 1.0 * 1024 / height
    intrinsics[1, :] *= scale_h

    return intrinsics, img


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected[2:3][K_xyz_reprojected[2:3] == 0] += 0.00001
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(args, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, scan):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = None
    masks = []
    s = s_all[scan]
    for i in range(s, 11):
        # mask = np.logical_and(dist < i / 4, relative_depth_diff < i / 1300)
        mask = np.logical_and(dist < i * args.dist_base, relative_depth_diff < i * args.rel_diff_base)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    scan = os.path.basename(scan_folder)
    s = s_all[scan]
    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # src_views = src_views[:args.num_view]
        # load the camera parameters
        ref_depth_max, ref_depth_min = read_depth_parameters(
                os.path.join(pair_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        if args.img_mode == "nresize":
            ref_intrinsics, ref_extrinsics = read_camera_parameters(
                os.path.join(pair_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
            ref_img = read_img(os.path.join(out_folder, 'images/{:0>8}.jpg'.format(ref_view)))
            flag_img = ref_img
            ref_intrinsics, _ = scale_input(ref_intrinsics, flag_img)
        else:
            ref_intrinsics, ref_extrinsics = read_camera_parameters(
                os.path.join(out_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
            # load the reference image
            ref_img = read_img(os.path.join(out_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        confidence2 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage2.pfm'.format(ref_view)))[0]
        confidence1 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage1.pfm'.format(ref_view)))[0]
        # photo_mask = np.logical_and(np.logical_and(np.logical_and(confidence > args.conf[3], confidence3 > args.conf[2]), confidence2 > args.conf[1]), confidence1 > args.conf[0])
        # photo_mask = np.logical_and(np.logical_and(np.logical_and(confidence > args.conf[3],confidence3 > calculate_percentile(confidence3, 15)),
        #                                            confidence2 > calculate_percentile(confidence2, 10)), confidence1 > calculate_percentile(confidence1, 5))
        photo_mask = np.logical_and(np.logical_and(confidence > calculate_percentile(confidence, args.conf[2]),confidence2 > calculate_percentile(confidence2, args.conf[1])),
                                                   confidence1 > calculate_percentile(confidence1, args.conf[0]))
        # flag_img = ref_img
        # confis.append(confidence)  # [0.05, 0.1, 0.15, 0.5]

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        dy_range = len(src_views) + 1
        geo_mask_sums = [0] * (dy_range - s)
        for src_view in src_views:
            # camera parameters of the source view
            if args.img_mode == "nresize":
                src_intrinsics, src_extrinsics = read_camera_parameters(
                    os.path.join(pair_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
                src_intrinsics, _ = scale_input(src_intrinsics, flag_img)
            else:
                src_intrinsics, src_extrinsics = read_camera_parameters(
                    os.path.join(out_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(args, ref_depth_est, ref_intrinsics,
                                                                                               ref_extrinsics, src_depth_est,
                                                                                               src_intrinsics, src_extrinsics, scan)

            geo_mask_sum += geo_mask.astype(np.int32)
            for i in range(s, dy_range):
                geo_mask_sums[i - s] += masks[i - s].astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        maskdepth = np.logical_and(depth_est_averaged >= ref_depth_min,
                                   depth_est_averaged <= ref_depth_max)
        # at least args.thres_view source views matched
        geo_mask = geo_mask_sum >= dy_range
        for i in range(s, dy_range):
            geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - s] >= i)

        final_mask = np.logical_and(photo_mask, geo_mask)
        final_mask = np.logical_and(final_mask, maskdepth)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.jpg".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.jpg".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.jpg".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset

        color = ref_img[valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan):
    scan_id = int(scan[4:])
    save_name = 'mvsnet{:0>3}_13.ply'.format(scan_id)

    pair_folder = os.path.join(args.testpath, args.split, scan)
    scan_folder = os.path.join(args.outdir, args.split, scan)
    out_folder = os.path.join(args.outdir, scan)

    args.dist_base = dist_all[scan]
    args.rel_diff_base = rel_diff_all[scan]
    filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.plydir, save_name))


def pcd_filter(testlist, number_worker):

    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        logger.info("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()


def initLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)
    logfile = os.path.join(args.logdir, 'fusion-' + curTime + '.log')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    if not args.nolog:
        fileHandler = logging.FileHandler(logfile, mode='a')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.info("Logger initialized.")
    logger.info("Writing logs to file: {}".format(logfile))
    logger.info("Current time: {}".format(curTime))

    return logger


if __name__ == '__main__':
    logger = initLogger()
    with open(os.path.join(args.testlist)) as f:
        testlist = [line.rstrip() for line in f.readlines()]
    if not os.path.isdir(args.plydir):
        os.mkdir(args.plydir)
    if args.single_processor:
        for scene in testlist:
            save_name = '{}.ply'.format(scene)

            pair_folder = os.path.join(args.testpath, args.split, scene)
            scan_folder = os.path.join(args.outdir, args.split, scene)
            out_folder = os.path.join(args.outdir, scene)

            if scene in tank_cfg.scenes:
                scene_cfg = getattr(tank_cfg, scene)
                args.conf = scene_cfg.conf
                args.dist_base = dist_all[scene]
                args.rel_diff_base = rel_diff_all[scene]
            filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.plydir, save_name))
    else:
        pcd_filter(testlist, args.num_worker)

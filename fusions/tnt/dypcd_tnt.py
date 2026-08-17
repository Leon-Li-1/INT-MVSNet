import os
import cv2
import signal
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool
from plyfile import PlyData, PlyElement
import argparse
import re, json  # NOQA

from sklearn.preprocessing import scale  # NOQA

parser = argparse.ArgumentParser()
parser.add_argument("--testpath", type=str, default="/home/lida/lida/2Ddataset/tankandtemples/")
parser.add_argument('--outdir', default='./outputs/tnt/jiayou_test', help='output dir')
parser.add_argument('--ply_path', type=str, default='outputs/tnt/jiayou_test/dypcd_fusion_plys_et')

parser.add_argument('--split', type=str, default='intermediate', choices=['intermediate', 'advanced'])
parser.add_argument('--list_file', type=str, default='./datasets/lists/tnt/intermediate.txt')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--single_processor', default=True, action='store_true')

parser.add_argument('--cam_mode', type=str, default='origin', choices=['origin', 'short_range'])
parser.add_argument('--img_mode', type=str, default='resize', choices=['resize', 'crop'])

parser.add_argument('--dist_base', type=float, default=1 / 4)
parser.add_argument('--rel_diff_base', type=float, default=1 / 1300)

args = parser.parse_args()
# 0.1,0.1,0.1,0其它都取值0.5
if args.split == 'intermediate':
    s_all = {'Family':2, 'Francis':5, 'Horse':2, 'Lighthouse':3, 'M60':3, 'Panther':2, 'Playground':3, 'Train':3}
    conf_all = {'Family':0.5, 'Francis':0.5, 'Horse':0.5, 'Lighthouse':0.6, 'M60':0.6, 'Panther':0.5, 'Playground':0.6, 'Train':0.5}
    dist_all = {'Family':1/12, 'Francis':1/8, 'Horse':1/2, 'Lighthouse':1/5, 'M60':1/8, 'Panther':1/5, 'Playground':1/5, 'Train':1/6}
    rel_diff_all = {'Family':1/1500, 'Francis':1/1500, 'Horse':1/1000, 'Lighthouse':1/1300, 'M60':1/1600, 'Panther':1/1500, 'Playground':1/1600, 'Train':1/1600}
# elif args.split == 'advanced':
#     s_all = {'Auditorium':2, 'Ballroom':2, 'Courtroom':1, 'Museum':1, 'Palace':1, 'Temple':1}
#     conf_all = {'Auditorium': 0.0, 'Ballroom': 0.5, 'Courtroom': 0.5, 'Museum': 0.6, 'Palace': 0.6, 'Temple': 0.5}
#     dist_all = {'Auditorium':1/5, 'Ballroom':1/5, 'Courtroom':1/5, 'Museum':1/5, 'Palace':1/5, 'Temple':1/5}
#     rel_diff_all = {'Auditorium':1/1300, 'Ballroom':1/1400, 'Courtroom':1/1400, 'Museum':1/1600, 'Palace':1/1600, 'Temple':1/1600}

# if args.split == 'intermediate':
#     s_all = {'Family':2, 'Francis':7, 'Horse':2, 'Lighthouse':3, 'M60':3, 'Panther':2, 'Playground':3, 'Train':3}
#     conf_all = {'Family':0.5, 'Francis':0.8, 'Horse':0.8, 'Lighthouse':0.8, 'M60':0.9, 'Panther':0.9, 'Playground':0.85, 'Train':0.9}
#     dist_all = {'Family':1/12, 'Francis':1/4, 'Horse':1/2, 'Lighthouse':1/4, 'M60':1/4, 'Panther':1/4, 'Playground':1/4, 'Train':1/4}
#     rel_diff_all = {'Family':1/1300, 'Francis':1/1300, 'Horse':1000, 'Lighthouse':1/1000, 'M60':1/1300, 'Panther':1/1300, 'Playground':1/1300, 'Train':1/1300}
elif args.split == 'advanced':
    s_all = {'Auditorium':3, 'Ballroom':3, 'Courtroom':3, 'Museum':3, 'Palace':3, 'Temple':3}
    conf_all = {'Auditorium': 0.7, 'Ballroom': 0.8, 'Courtroom': 0.8, 'Museum': 0.8, 'Palace': 0.9, 'Temple': 0.8}
    dist_all = {'Auditorium':2.0, 'Ballroom':1.5, 'Courtroom':1.5, 'Museum':1.5, 'Palace':1.5, 'Temple':1.5}
    rel_diff_all = {'Auditorium':2/1000, 'Ballroom':1/1000, 'Courtroom':2/1000, 'Museum':1/1000, 'Palace':1/1000, 'Temple':2/1000}


# if args.split == 'intermediate':
#     s_all = {'Family':2, 'Francis':9, 'Horse':2, 'Lighthouse':3, 'M60':3, 'Panther':2, 'Playground':3, 'Train':3}
#     conf_all = {'Family':0.3, 'Francis':0.2, 'Horse':0.0, 'Lighthouse':0.6, 'M60':0.6, 'Panther':0.5, 'Playground':0.6, 'Train':0.5}
#     dist_all = {'Family':1/12, 'Francis':1/8, 'Horse':1/4, 'Lighthouse':1/5, 'M60':1/8, 'Panther':1/5, 'Playground':1/5, 'Train':1/6}
#     rel_diff_all = {'Family':1/1600, 'Francis':1/1600, 'Horse':1/1300, 'Lighthouse':1/1300, 'M60':1/1600, 'Panther':1/1500, 'Playground':1/1600, 'Train':1/1600}
# elif args.split == 'advanced':
#     s_all = {'Auditorium':2, 'Ballroom':2, 'Courtroom':1, 'Museum':1, 'Palace':1, 'Temple':1}
#     conf_all = {'Auditorium': 0.5, 'Ballroom': 0.6, 'Courtroom': 0.6, 'Museum': 0.6, 'Palace': 0.5, 'Temple': 0.5}
#     dist_all = {'Auditorium':1/5, 'Ballroom':1/5, 'Courtroom':1/5, 'Museum':1/5, 'Palace':1/5, 'Temple':1/5}
#     rel_diff_all = {'Auditorium':1/1300, 'Ballroom':1/1400, 'Courtroom':1/1400, 'Museum':1/1600, 'Palace':1/1600, 'Temple':1/1600}


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale1 = None
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

    scale1 = float(file.readline().rstrip())
    if scale1 < 0:  # little-endian
        endian = '<'
        scale1 = -scale1
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale1


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


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read ref_depth_max, ref_depth_min
def read_depth_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix

    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[-1])
    return depth_max, depth_min


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


# project the reference point cloud into the source view, then project back
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


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, scan):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    s = s_all[scan]
    mask = None
    masks = []
    dist_base = dist_all[scan]
    rel_diff_base = rel_diff_all[scan]
    for i in range(s, 11):
        mask = np.logical_and(dist < i * dist_base, relative_depth_diff < i * rel_diff_base)
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

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        ref_depth_max, ref_depth_min = read_depth_parameters(
            os.path.join(pair_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(out_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(out_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        conf_thresh = conf_all[scan]
        photo_mask = confidence > conf_thresh
        # add by liyi 2 lines
        # flag_img = ref_img

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
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(out_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics,
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
        # at least thres_view source views matched
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

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset

        color = ref_img[valid_points]
        # color = ref_img[28:1080 - 28, :, :][valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

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


def dypcd_filter_worker(scene):
    save_name = '{}.ply'.format(scene)
    pair_folder = os.path.join(args.testpath, args.split, scene)
    scan_folder = os.path.join(args.outdir, scene)
    out_folder = os.path.join(args.outdir, scene)
    filter_depth(scene, pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':

    with open(os.path.join(args.list_file)) as f:
        testlist = [line.rstrip() for line in f.readlines()]

    if not os.path.isdir(args.ply_path):
        os.mkdir(args.ply_path)
    if args.single_processor:
        for scene in testlist:
            save_name = '{}.ply'.format(scene)
            pair_folder = os.path.join(args.testpath, args.split, scene)
            scan_folder = os.path.join(args.outdir, scene)
            out_folder = os.path.join(args.outdir, scene)
            filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.ply_path, save_name))
    else:
        partial_func = partial(dypcd_filter_worker)
        p = Pool(args.num_workers, init_worker)
        try:
            p.map(partial_func, testlist)
        except KeyboardInterrupt:
            print("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()

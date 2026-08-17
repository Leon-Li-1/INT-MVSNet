# -*- coding: utf-8 -*-
# @Description: Options settings & configurations for MambaMVSNet.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07

import argparse


def get_opts():
    parser = argparse.ArgumentParser(description="args")

    # network
    parser.add_argument('--num_stage', type=int, default=3)
    parser.add_argument('--base_channels', type=int, default=8)
    parser.add_argument("--sample1", type=dict, nargs='+', default={"num_hypotheses": 48, "interval_ratio": 4})
    parser.add_argument("--sample2", type=dict, nargs='+', default={"num_hypotheses": 32, "interval_ratio": 2})
    parser.add_argument("--sample3", type=dict, nargs='+', default={"num_hypotheses": 8, "interval_ratio": 1})
    parser.add_argument("--group", type=int, default=8)

    # dataset
    parser.add_argument("--img_size", type=int, nargs='+', default=[1184, 1600])  # dtu: 1184, 1600  tnt:1024, -1
    parser.add_argument("--datapath", default="/home/lida/lida/2Ddataset/dtu_data/DTU/dtu_test", type=str)  # dtu_training dtu_test
    parser.add_argument("--trainlist", default="datasets/lists/dtu/train.txt", type=str)
    parser.add_argument("--testlist", default="datasets/lists/dtu/test_copy.txt", type=str)
    # parser.add_argument('--datapath', default="/home/lida/lida/2Ddataset/tankandtemples", help='data path for training')
    # parser.add_argument('--trainlist', default="./datasets/lists/blendedmvs/low_res_all.txt", help='data list for training')
    # parser.add_argument('--testlist', default="./datasets/lists/dtu/intermediate.txt", help='data list for testing')
    parser.add_argument("--dataset_name", type=str, default="general_eval", choices=["general_eval, tnt_eval"])
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
    parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')
    parser.add_argument("--nviews", type=int, default=5)
    parser.add_argument("--inverse_depth", default=False, help="for tnt?")
    # training and val
    parser.add_argument("--val", action="store_true")
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--scheduler', type=str, default="cosinelr", choices=["steplr", "cosinelr"])
    parser.add_argument('--warmup', type=float, default=0.33, help='warmup epochs')
    parser.add_argument('--milestones', type=float, nargs='+', default=[10, 12, 14], help='lr schedule')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay at every milestone')
    parser.add_argument('--resume', default=False, help='if resume')
    parser.add_argument('--start_ckpts', type=str, default=None, help='path to the resume model')
    parser.add_argument('--log_dir', type=str, default="./outputs/tnt/jiayou_test", help='path to the log dir')
    parser.add_argument('--dlossw', type=float, nargs='+', default=[0.5, 1.0, 2.0], help='depth loss weight for different stage')
    parser.add_argument('--refine', default=True, help='if ray encoding')

    # loss weights
    parser.add_argument('--wrecon', type=float, default=8.0)
    parser.add_argument('--wssim', type=float, default=6.0)
    parser.add_argument('--wsmooth', type=float, default=0.18)
    parser.add_argument('--w_icc', type=float, default=0.01)
    parser.add_argument('--max_w_icc', type=float, default=0.32)
    parser.add_argument('--w_scc', type=float, default=0.01)
    parser.add_argument('--mask_conf', type=float, default=0.95)
    parser.add_argument('--p_icc', type=float, default=0.1)

    # log
    parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
    parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')

    # testing
    parser.add_argument("--test", default=True, help='set test')
    parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
    parser.add_argument('--outdir', default="./outputs/dtu/jiayou_test", help='output dir')
    parser.add_argument('--split', type=str, default='intermediate', choices=['intermediate', 'advanced'], help='intermediate|advanced for tanksandtemples')
    parser.add_argument('--img_mode', type=str, default='resize', choices=['resize', 'crop'], help='image resolution matching strategy for TNT dataset')
    parser.add_argument('--cam_mode', type=str, default='origin', choices=['origin', 'short_range'], help='camera parameter strategy for TNT dataset')
    parser.add_argument('--nolog', action='store_true', help='do not log into .log file')
    parser.add_argument('--display', default=True, action='store_true', help='continue to train the model')

    # pcd
    parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
    parser.add_argument('--filter_method', type=str, default='pcd', help="filter method")

    parser.add_argument('--conf', type=float, default=0.8, help='prob confidence, for pcd')
    parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view, for pcd')
    parser.add_argument('--depth_thres', type=float, default=0.001, help='depth_thres for pcd')
    parser.add_argument('--img_dist_thres', type=float, default=0.75, help='depth_thres for pcd')

    # device and distributed
    parser.add_argument("--no_cuda", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync_bn", default=False)

    return parser.parse_args()

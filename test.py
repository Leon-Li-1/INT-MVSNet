import os, time, sys, gc, cv2, logging, errno  # NOQA
import numpy as np
import torch
import torch.nn as nn  # NOQA
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tools import *

from datasets.data_io import *
from datasets.general_eval import MVSDataset as DTUDataset
from datasets.general_eval_tnt import MVSDataset as TNTDataset

from networks.mvsnet import MVSNet
from networks.utils import *
from networks.utils.opts import get_opts


cudnn.benchmark = True

args = get_opts()


def test():
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    total_time = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)
            start_time = time.time()
            # @Note MambaMVSNet main
            outputs = model(sample_cuda, "test")
            end_time = time.time()
            total_time += end_time - start_time
            outputs = tensor2numpy_str(outputs)
            del sample_cuda

            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(args.num_stage)].numpy()
            imgs = sample["imgs"]
            logger.info('Iter {}/{}, Time:{:.3f} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            for filename, cam, img, depth_est, depth2, depth1, photometric_confidence, photometric_confidence2, photometric_confidence1 in zip(filenames,
                                                                                                                                               cams, imgs, outputs["depth"],
                                                                                                                                               outputs["stage2"]["depth"],
                                                                                                                                               outputs["stage1"]["depth"],
                                                                                                                                               outputs["photometric_confidence"],
                                                                                                                                               outputs["stage2"]["photometric_confidence"],
                                                                                                                                               outputs["stage1"]["photometric_confidence"]):

                h, w = photometric_confidence.shape
                img = img[0]
                img = inv_normalize(img).numpy()
                cam = cam[0]
                photometric_confidence2 = cv2.resize(photometric_confidence2, (w, h), interpolation=cv2.INTER_NEAREST)
                photometric_confidence1 = cv2.resize(photometric_confidence1, (w, h), interpolation=cv2.INTER_NEAREST)
                confidence_filename2 = os.path.join(args.outdir, filename.format('confidence', '_stage2.pfm'))
                confidence_filename1 = os.path.join(args.outdir, filename.format('confidence', '_stage1.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                depth_filename2 = os.path.join(args.outdir, filename.format('depth_est', '_stage2.pfm'))
                depth_filename1 = os.path.join(args.outdir, filename.format('depth_est', '_stage1.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                # save depth maps
                save_pfm(depth_filename, depth_est)
                save_pfm(depth_filename2, depth2)
                save_pfm(depth_filename1, depth1)
                # save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                save_pfm(confidence_filename2, photometric_confidence2)
                save_pfm(confidence_filename1, photometric_confidence1)
                # save cams, img

                if args.display:
                    depth_color = visualize_depth(depth_est)
                    cv2.imwrite(os.path.join(args.outdir, filename.format('depth_est', '.jpg')), depth_color)
                    # save confidence maps
                    cv2.imwrite(os.path.join(args.outdir, filename.format('confidence', '.jpg')), visualize_depth(photometric_confidence))
                # inter_val = outputs["stage4"]["interval"]
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

    torch.cuda.empty_cache()
    gc.collect()
    return total_time, len(TestImgLoader)


def initLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    curTime = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))

    if args.dataset_name == 'tnt_eval':
        logfile = os.path.join(args.log_dir, 'TNT-test-' + curTime + '.log')
    elif args.dataset_name == 'general_eval':
        logfile = os.path.join(args.log_dir, 'test-' + curTime + '.log')
    else:
        raise NotImplementedError("Don't support dataset: {}".format(args.dataset_name))
    # add by liyi, used for creat logfile
    if not os.path.exists(os.path.dirname(logfile)):
        try:
            os.makedirs(os.path.dirname(logfile))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
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

    settings_str = "All settings:\n"
    for k, v in vars(args).items():
        settings_str += '{0}: {1}\n'.format(k, v)
    logger.info(settings_str)

    return logger


if __name__ == '__main__':
    logger = initLogger()

    # dataset, dataloader
    if args.dataset_name == 'general_eval':
        test_dataset = DTUDataset(args, args.testlist, "test")
    else:
        raise NotImplementedError("Don't support dataset: {}".format(args.dataset_name))
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # @Note MambaMVSNet model
    model = MVSNet(args)
    logger.info("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(state_dict['model'], strict=False)

    model.cuda()
    model.eval()

    test()

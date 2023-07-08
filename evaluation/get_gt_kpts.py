import time
import warnings
# import tensorflow
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import sys
sys.path.append('.')
from config.config_reader import parse_args, create_parser

from load_dataloader import load_dataloader

from loss.compute_loss import *
# import visualize

# from utils import Logger, mkdir_p #, save_images, save_3d_images, save_multi_images,save_images_2
from utils.model_utils import *
import cv2
import numpy as np
import copy




def main():

    args = parse_args(create_parser())

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


    
    
    
def main_worker(gpu, ngpus_per_node, args, mode):
    torch.cuda.set_device(args.gpu)
    
    # Data loading code
    root = os.path.join(args.data)
    

    dataset= load_dataloader(args, mode)
    train_sampler = None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    all_kpts = validate(loader, args)
    return all_kpts
#############################

def validate(loader, args):
    all_kpts = []
    valid_joints = (3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19) + (14,)
    all_3d_kpts = []
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, all_cam_items in enumerate(loader):
            all_cam_gt0 = []
            all_cam_gt1 = []
            
            for cam_num in range(len(all_cam_items['image'])):
                _, gt_1 = all_cam_items['ground_truth'][cam_num]
                #gt_0 = gt_0.cuda(args.gpu, non_blocking=True)
                gt_1 = gt_1.cuda(args.gpu, non_blocking=True)
                #all_cam_gt0.append(gt_0)
                all_cam_gt1.append(gt_1)

            intrinsics = all_cam_items['calib_intrinsics']
            extrinsics = all_cam_items['calib_extrinsics']
            distortions = all_cam_items['calib_distortions']
            #extrinsics is [batch, view, 4,4]
            
            ### all views unproject to same world 
            p3d = all_cam_gt1[0]
            homog_shape = list(p3d.shape)
            homog_shape[2] +=1
            full_points_homog = torch.ones(homog_shape).cuda(0)
            full_points_homog[:, :,:3] = p3d.cuda(0)
            #use view 0
            world_3d = torch.matmul(torch.inverse(extrinsics[:, 0, :,:].cuda(0)), torch.permute(full_points_homog, (0,2,1)))
            world_3d = torch.permute(world_3d, (0,2,1))[:,valid_joints,:]
            
            world_3d = world_3d[:,:,:3]/world_3d[:,:,[3]]
            
            all_kpts.append(world_3d)


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
   
    all_kpts = torch.cat(all_kpts, 0)
    
    return all_kpts


if __name__ == '__main__':
    main()

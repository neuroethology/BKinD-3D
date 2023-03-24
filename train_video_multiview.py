import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from config.config_reader import parse_args, create_parser

from dataloader.load_dataloader import load_dataloader

from model.unsupervised_model_seg_vol_edge_tree import Model

from loss.compute_loss import *

from utils import Logger, mkdir_p, save_images, save_3d_images, save_multi_images,save_images_2
from utils.model_utils import *
from utils.triangulation import Camera
import cv2
import numpy as np
import copy

import gc

best_loss = 10000

def main():

    args = parse_args(create_parser())

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    output_shape = (int(args.image_size/4), int(args.image_size/4))
    model = Model(args.nkpts, output_shape=output_shape,
                  volume_size=args.volume_size,
                  cuboid_side=args.cuboid_side,
                  v2v_features=args.v2v_features)


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    loss_module = computeLoss(args)
    loss_3d = nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    num_edges = args.nkpts*(args.nkpts-1)//2
    edge_weights = torch.randn(num_edges, requires_grad=True, device = torch.device('cuda'))
    optimizer_2 = torch.optim.SGD([edge_weights], args.lr*512,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    running_average = torch.zeros(num_edges, requires_grad=False, device = torch.device('cuda'))


    # optionally resume from a checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    title = 'Landmark-discovery'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']

            edge_weights = checkpoint['edge_weights']
            optimizer_2.load_state_dict(checkpoint['optimizer_2'])

            running_average = checkpoint['running_average']

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss',])

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data)
    valdir = os.path.join(args.data)

    train_dataset, val_dataset = load_dataloader(args)
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    if args.evaluate:
        validate(val_loader, model, loss_module, 0, args)
        return

    is_best = True
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, running_average = train(train_loader, model, edge_weights, running_average, loss_module, loss_3d, optimizer, optimizer_2, epoch, args)

        # evaluate on validation set every val_schedule epochs
        if epoch > 0 and epoch%args.val_schedule == 0:
            test_loss = validate(val_loader, model, loss_module, epoch, args)
        else:
            test_loss = 10000  # set test_loss = 100000 when not using validate

        logger.append([args.lr * (0.1 ** (epoch // args.schedule)), train_loss, test_loss])

        # remember best acc@1 and save checkpoint
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
            'edge_weights': edge_weights,
            'optimizer_2' : optimizer_2.state_dict(),    
            'running_average': running_average        
        }, is_best, checkpoint=args.checkpoint)


#############################

def train(train_loader, model, edge_weights, running_average, loss_module, loss_3d, optimizer,optimizer_2, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i, all_cam_items in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Compute reconstruction loss over each camera view
        all_cam_inputs = []
        all_cam_tr_inputs = []
        all_cams = []

        # all_cam_conf = []
        # all_cam_conf_ori = []

        # all_cam_recon = []
        # all_cam_heatmap = []

        # loss = 0

        rot_im1_list = []
        rot_im2_list = []
        rot_im3_list = []


        for cam_num in range(len(all_cam_items['image'])):
            inputs, tr_inputs = all_cam_items['image'][cam_num]
            loss_mask, in_mask = all_cam_items['mask'][cam_num]
            # tr versions of rotated images
            rot_im1, rot_im2, rot_im3 = all_cam_items['rotation'][cam_num]

            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                tr_inputs = tr_inputs.cuda(args.gpu, non_blocking=True)
                loss_mask = loss_mask.cuda(args.gpu, non_blocking=True)
                in_mask = in_mask.cuda(args.gpu, non_blocking=True)
                rot_im1 = rot_im1.cuda(args.gpu, non_blocking=True)
                rot_im2 = rot_im2.cuda(args.gpu, non_blocking=True)               
                rot_im3 = rot_im3.cuda(args.gpu, non_blocking=True)

            all_cam_inputs.append(inputs)
            all_cam_tr_inputs.append(tr_inputs)
            rot_im1_list.append(rot_im1)
            rot_im2_list.append(rot_im2)
            rot_im3_list.append(rot_im3)                        

        intrinsics = all_cam_items['calib_intrinsics']
        extrinsics = all_cam_items['calib_extrinsics']
        distortions = all_cam_items['calib_distortions']

        # print(distortions)
        # error

        # proj_matrices =  shape (batch_size, n_views, 3, 4)
        # Make Camera Classes
        # batch x cam_num x 4 x 4
        #print(extrinsics.size())
        for view in range(extrinsics.size()[1]):
            curr_batch = []
            for batch in range(extrinsics.size()[0]):
                curr_batch.append(Camera(extrinsics[batch, view, :3, :3], extrinsics[batch, view, :3, 3], 
                    intrinsics[batch, view], distortions[batch, view]))

            all_cams.append(curr_batch)

        output = model(all_cam_inputs, all_cam_tr_inputs, edge_weights, all_cams, all_cam_items)

        all_cam_points = [torch.stack(item, axis=2) for item in output['tr_pos']]
        all_cam_points_ori = [torch.stack(item, axis=2) for item in output['pos']]


        if epoch >= args.curriculum:
            ori_loss, length_loss, ssim_list, running_average = loss_module.update_loss_4(all_cam_inputs, all_cam_tr_inputs, running_average, output['samples'], loss_mask, output, epoch, args.nkpts)    

            loss = ori_loss + length_loss/1000000
        else:
            ori_loss, ssim_list = loss_module.update_loss_2(all_cam_inputs, all_cam_tr_inputs, loss_mask, output, epoch)    
            loss = ori_loss

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        optimizer_2.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        gc.collect()

        if i % args.print_freq == 0:
            progress.display(i)
            
            if args.visualize:

                # print(all_cam_points)
                save_multi_images(all_cam_items['image'], all_cam_points_ori,
                    all_cam_points, output['recon'], ssim_list,
                    output['tr_kpt_out'], output['tr_kpt_cond'], 
                    [output['tr_confidence'] for _ in range(len(all_cam_items['image']))], epoch, args, epoch)
                # print(output['tr_pos'][0].size(), projected_points[:, :, 0].size())
                # save_3d_images(pts_3d.detach().cpu(), epoch, args, epoch)


                # save_images_2(tr_inputs, output, epoch, args, epoch)

                # if epoch >= args.curriculum:

                #     output['recon'] = output_mv['recon']
                #     output['tr_pos'] = (projected_points[:, -1, :, 0], projected_points[:, -1, :, 1])
                #     save_images(tr_inputs, output, epoch, args, str(epoch) + 'projected')

    return losses.avg, running_average

def validate(val_loader, model, loss_module, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, all_cam_items in enumerate(val_loader):

            # Compute reconstruction loss over each camera view
            for cam_num in range(len(all_cam_items['image'])):
                inputs, tr_inputs = all_cam_items['image'][cam_num]
                loss_mask, in_mask = all_cam_items['mask'][cam_num]
                rot_im1, rot_im2, rot_im3 = all_cam_items['rotation'][cam_num]

                if args.gpu is not None:
                    inputs = inputs.cuda(args.gpu, non_blocking=True)
                    tr_inputs = tr_inputs.cuda(args.gpu, non_blocking=True)
                    loss_mask = loss_mask.cuda(args.gpu, non_blocking=True)
                    in_mask = in_mask.cuda(args.gpu, non_blocking=True)
                    rot_im1 = rot_im1.cuda(args.gpu, non_blocking=True)
                    rot_im2 = rot_im2.cuda(args.gpu, non_blocking=True)               
                    rot_im3 = rot_im3.cuda(args.gpu, non_blocking=True)

                if epoch < args.curriculum:
                    output = model(inputs, tr_inputs)
                else:
                    output = model(inputs, tr_inputs, gmtr_x1 = rot_im1, gmtr_x2 = rot_im2, gmtr_x3 = rot_im3)

                loss = loss_module.update_loss(inputs, tr_inputs, loss_mask, output, epoch)
                
                # measure accuracy and record loss
                losses.update(loss.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return losses.avg


if __name__ == '__main__':
    main()

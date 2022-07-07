import time
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from config.config_reader import parse_args, create_parser

from dataloader.load_dataloader import load_dataloader

from model.unsupervised_model import Model

from loss.compute_loss import *

from utils import Logger, mkdir_p, save_images
from utils.model_utils import *
import cv2
import numpy as np
import copy


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
    model = Model(args.nkpts, output_shape=output_shape)

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
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, loss_module, 0, args)
        return

    is_best = True
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss = train(train_loader, model, loss_module, loss_3d, optimizer, epoch, args)

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
        }, is_best, checkpoint=args.checkpoint)

#############################

def undistort_torch(points, matrix, dist):
    mat = torch.linalg.inv(matrix)
    pts = torch.linalg.matmul(points, mat[:2,:2]) + mat[:2, 2]

    # translated from OpenCV undistortPoints code
    x0 = x = pts[:, 0]
    y0 = y = pts[:, 1]
    k = dist

    ITERS = 5
    for j in range(ITERS):
        r2 = torch.square(x) + torch.square(y)
        icdist = 1/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2)
        deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x)
        deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y
        x = (x0 - deltaX)*icdist
        y = (y0 - deltaY)*icdist

    return torch.stack([x, y]).T

def distort_torch(points, matrix, dist):
    x = points[:, 0]
    y = points[:, 1]
    k = dist
    r2 = torch.square(x) + torch.square(y)
    x_new = x * (1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2) + 2*k[2]*x*y + k[3]*(r2 + 2*x*x)
    y_new = y * (1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2) + 2*k[3]*x*y + k[2]*(r2 + 2*y*y)
    pts = torch.stack([x_new, y_new]).T
    pts = torch.linalg.matmul(pts, matrix[:2,:2]) + matrix[:2, 2]
    return pts

def triangulate_simple(points, camera_mats, conf):
    num_cams = len(camera_mats)
    A = torch.zeros((num_cams * 2, 4)).to(points.device)
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]

    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

def project_points(points_3d, camera_params):
    extrinsics = camera_params['extrinsics']
    intrinsics = camera_params['intrinsics']
    distortions = camera_params['distortions']
    proj_2d = []
    for i in range(len(intrinsics)):
        proj_und = torch.matmul(points_3d, extrinsics[i][:3,:3].T) + extrinsics[i][:3, 3]
        pts = proj_und[:, :2] / proj_und[:,2:]
        p = distort_torch(pts, intrinsics[i], distortions[i])
        proj_2d.append(p)
    proj_2d = torch.stack(proj_2d)
    return proj_2d

def triangulate_points_full(points_2d, camera_params, confidence):
    extrinsics = camera_params['extrinsics']
    intrinsics = camera_params['intrinsics']
    distortions = camera_params['distortions']
    pts_2d_und = torch.stack(
        [undistort_torch(points_2d[i], intrinsics[i], distortions[i])
         for i in range(len(intrinsics))])
    pts_3d = torch.stack([triangulate_simple(pts_2d_und[:, i], extrinsics, confidence[:, i])
                          for i in range(pts_2d_und.shape[1])])
    return pts_3d


#############################

def train(train_loader, model, loss_module, loss_3d, optimizer, epoch, args):

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
        all_cam_points = []
        all_cam_points_ori = []

        all_cam_conf = []
        all_cam_conf_ori = []

        loss = 0

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

            all_cam_points.append(torch.stack(output['tr_pos'], axis=2))
            all_cam_points_ori.append(torch.stack(output['pos'], axis=2))

            all_cam_conf.append(output['tr_confidence'])
            all_cam_conf_ori.append(output['confidence'])

            loss += loss_module.update_loss(inputs, tr_inputs, loss_mask, output, epoch)
            

        ############################### Map to 3D (Loop through batch)
        # First dimension is the batch
        if epoch >= args.curriculum:

            full_points = (torch.stack(all_cam_points, dim = 1)+1)*500
            full_points_ori = (torch.stack(all_cam_points_ori, dim = 1)+1)*500


            full_conf = torch.stack(all_cam_conf, dim = 1)
            full_conf_ori = torch.stack(all_cam_conf_ori, dim = 1)

            intrinsics = all_cam_items['calib_intrinsics']
            extrinsics = all_cam_items['calib_extrinsics']
            distortions = all_cam_items['calib_distortions']


            if args.gpu is not None:
                intrinsics = intrinsics.cuda(args.gpu, non_blocking=True)
                extrinsics = extrinsics.cuda(args.gpu, non_blocking=True)
                distortions = distortions.cuda(args.gpu, non_blocking=True)


            projected_points = []
            projected_points_ori = []
            for b in range(full_points.size()[0]):

                camera_params = {
                    'extrinsics': extrinsics[b],
                    'intrinsics': intrinsics[b],
                    'distortions': distortions[b]
                }

                pts_3d = triangulate_points_full(full_points[b], camera_params, full_conf[b])

                # reproject
                proj_2d = project_points(pts_3d, camera_params)

                projected_points.append(proj_2d/500 - 1)


                pts_3d_ori = triangulate_points_full(full_points_ori[b], camera_params, full_conf_ori[b])
                # reproject
                proj_2d_ori = project_points(pts_3d_ori, camera_params)

                projected_points_ori.append(proj_2d_ori/500 - 1)


            projected_points = torch.stack(projected_points, dim = 0).clip(-1, 1)
            projected_points_ori = torch.stack(projected_points_ori, dim = 0).clip(-1, 1)

            tester = (full_points/500-1).clone().detach()
            loss2 = (torch.sum((projected_points - tester) ** 2, dim = -1)).mean()*10
            loss += loss2
            print("Reprojection: " + str(loss2))

            ##########################################
            # Multiview reconstruction loss
            # TODO: Also add rotation loss here?
            for cam_num in range(len(all_cam_items['image'])):
                inputs, tr_inputs = all_cam_items['image'][cam_num]
                loss_mask, in_mask = all_cam_items['mask'][cam_num]

                if args.gpu is not None:
                    inputs = inputs.cuda(args.gpu, non_blocking=True)
                    tr_inputs = tr_inputs.cuda(args.gpu, non_blocking=True)
                    loss_mask = loss_mask.cuda(args.gpu, non_blocking=True)
                    in_mask = in_mask.cuda(args.gpu, non_blocking=True)
                
                output_mv = model.cross_view_recon(inputs, projected_points_ori[:, cam_num, :, 0],
                    projected_points_ori[:, cam_num, :, 1], projected_points[:, cam_num, :, 0],
                    projected_points[:, cam_num, :, 1])

                loss3 = loss_module.update_loss(inputs, tr_inputs, loss_mask, output_mv, epoch)
                loss += loss3

            print("Reprojection reconstruction: " + str(loss3))


        ##########################################

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
            if args.visualize:
                # print(output['tr_pos'][0].size(), projected_points[:, :, 0].size())
                save_images(tr_inputs, output, epoch, args, epoch)

                if epoch >= args.curriculum:

                    output['recon'] = output_mv['recon']
                    output['tr_pos'] = (projected_points[:, -1, :, 0], projected_points[:, -1, :, 1])
                    save_images(tr_inputs, output, epoch, args, str(epoch) + 'projected')

    return losses.avg

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

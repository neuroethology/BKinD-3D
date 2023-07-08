from __future__ import print_function, absolute_import

import os
import torchvision.transforms as transforms
import sys
sys.path.append('/home/amildravid/BKinD-main/2volume_edge')

import train_regress_h36m as h36m
import data_utils

def load_dataloader(args, mode):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    loader = data_utils.box_loader if args.bounding_box else data_utils.default_loader

    if args.dataset == 'H36M':
        root = os.path.join(args.data)
#         valdir = os.path.join(args.data)

        dataset = h36m.H36MDataset(mode, root, transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              target_transform=transforms.Compose([
                              transforms.Resize(args.image_size),
                              transforms.CenterCrop(args.image_size),
                              transforms.ToTensor(),
                              normalize,]),
                              image_size=[args.image_size, args.image_size],
                              loader=loader, frame_gap=args.frame_gap, crop_box=args.bounding_box)

#         val_dataset = h36m.H36MDataset(valdir, transforms.Compose([
#                               transforms.Resize(args.image_size),
#                               transforms.CenterCrop(args.image_size),
#                               transforms.ToTensor(),
#                               normalize,]),
#                               target_transform=transforms.Compose([
#                               transforms.Resize(args.image_size),
#                               transforms.CenterCrop(args.image_size),
#                               transforms.ToTensor(),
#                               normalize,]),
#                               image_size=[args.image_size, args.image_size],
#                               loader=loader, frame_gap=args.frame_gap, crop_box=args.bounding_box)

   

    return dataset

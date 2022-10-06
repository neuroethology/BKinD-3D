from __future__ import print_function, absolute_import

import os
import os.path

import numpy as np

import torch
import torch.utils.data as data
import torchvision

import torchvision.transforms.functional as TF

import h5py

from dataloader.data_utils import *

import cv2

import toml
from utils.triangulation import load_calibration
import re
from collections import defaultdict
from glob import glob
from tqdm import tqdm
from PIL import Image


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def true_basename(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename

def get_cam_name(config, fname):
    basename = true_basename(fname)

    cam_regex = config['triangulation']['cam_regex']
    match = re.search(cam_regex, basename)

    if not match:
        return None
    else:
        name = match.groups()[0]
        return name.strip()

def get_video_name(config, fname):
    basename = true_basename(fname)

    cam_regex = config['triangulation']['cam_regex']
    vidname = re.sub(cam_regex, '', basename)
    return vidname.strip()

def generate_pair_images(root, flies, gap=20, downsample=20):
    root = os.path.abspath(os.path.expanduser(root))

    # load the calibration
    calib_fname = os.path.join(root, 'Calibration', 'calibration.toml')
    calibration = load_calibration(calib_fname)

    # account for video cropping
    config_fname = os.path.join(root, 'config.toml')
    config = toml.load(config_fname)
    for i, cname in enumerate(calibration['names']):
        offset = config['cameras'][cname]['offset']
        calibration['intrinsics'][i, 0, 2] -= offset[0] # x
        calibration['intrinsics'][i, 1, 2] -= offset[1] # y

    outs = []

    for fly in flies:
        video_files = glob(os.path.join(root, fly, 'videos-raw-compressed', '*.mp4'))
        cam_videos = defaultdict(list)
        for vf in video_files:
            name = get_video_name(config, vf)
            cam_videos[name].append(vf)

        vid_names = cam_videos.keys()
        vid_names = sorted(vid_names, key=natural_keys)

        for name in tqdm(vid_names, ncols=70, desc=fly):
            fnames = cam_videos[name]
            cam_names = [get_cam_name(config, f) for f in fnames]
            fname_dict = dict(zip(cam_names, fnames))

            calib_temp = {'names': [],
                          'intrinsics': [],
                          'extrinsics': [],
                          'distortions': []}

            video_images = []
            video_images_next = []
            for ix_calib, cname in enumerate(calibration['names']):
                if cname in fname_dict:
                    calib_temp['names'].append(cname)
                    calib_temp['intrinsics'].append(calibration['intrinsics'][ix_calib])
                    calib_temp['extrinsics'].append(calibration['extrinsics'][ix_calib])
                    calib_temp['distortions'].append(calibration['distortions'][ix_calib])

                    images = []
                    images_next = []
                    cap = cv2.VideoCapture(fname_dict[cname])
                    framenum = 0
                    while True:
                        ret, img_bgr = cap.read()
                        if not ret: break
                        if framenum % downsample == 0:
                            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            images.append(img)
                        if framenum >= gap and (framenum - gap) % downsample == 0:
                            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            images_next.append(img)
                        framenum += 1
                    video_images.append(images)
                    video_images_next.append(images_next)
                    cap.release()

            n_frames = min([len(x) for x in video_images] +
                           [len(x) for x in video_images_next])
            n_cams = len(video_images)

            for ix_frame in range(n_frames):
                allcams_item = []
                for ix_cam in range(n_cams):
                    im0 = video_images[ix_cam][ix_frame]
                    im1 = video_images_next[ix_cam][ix_frame]
                    allcams_item.append([im0, im1])
                outs.append({'calibration': calib_temp,
                             'items': allcams_item})

    return outs


def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(np.array(rvec))
    out[:3,:3] = rotmat
    out[:3, 3] = np.array(tvec).flatten()
    out[3, 3] = 1
    return out



class FlyDataset(data.Dataset):
    """DataLoader for the fly dataset
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=box_loader, image_size=[128, 128],
                 simplified=False, crop_box=True, frame_gap=20):

        flies = ['Fly 1_0', 'Fly 2_0', 'Fly 3_0'] # eval on flies 4 and 5

        samples = generate_pair_images(root, flies, gap=frame_gap)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.crop_box = crop_box

        # Parameters for transformation
        self._image_size = image_size


    def __getitem__(self, index):

        image_dict = self.samples[index]

        # Assume same number of cameras for all samples.
        num_cameras = len(image_dict['items'])

        all_cam_items = {
            'image' : [],
            'mask' : [],
            'rotation' : [],
            'image_path' : [],
            'calibration': []
        }

        calib = image_dict['calibration']
        intrinsics = torch.stack(calib['intrinsics']).clone()

        for i in range(num_cameras):
            im0_arr, im1_arr = image_dict['items'][i]

            image0 = Image.fromarray(im0_arr)
            image1 = Image.fromarray(im1_arr)

            height, width = self._image_size[:2]

            ratio = min(image0.height / height, image0.width / width)
            crop_size = (int(ratio * height), int(ratio * width))
            crop_params = transforms.RandomCrop.get_params(image0, crop_size)

            # adjust intrinsics for crop
            intrinsics[i, :2, 0] -= crop_params[1] # x
            intrinsics[i, :2, 1] -= crop_params[0] # y

            image0 = TF.crop(image0, *crop_params)
            image1 = TF.crop(image1, *crop_params)

            # adjust intrinsics for resize
            ratio = self._image_size[0] / image0.height
            intrinsics[i, :2] *= ratio

            image0 = TF.resize(image0, self._image_size)
            image1 = TF.resize(image1, self._image_size)

            # Create 3 rotations
            deg = 90
            rot_image1 = TF.rotate(image1, deg)
            rot_image1 = self.target_transform(rot_image1)

            deg = 180
            rot_image2 = TF.rotate(image1, deg)
            rot_image2 = self.target_transform(rot_image2)

            deg = -90
            rot_image3 = TF.rotate(image1, deg)
            rot_image3 = self.target_transform(rot_image3)

            if self.transform is not None:
                image0 = self.transform(image0)
            if self.target_transform is not None:
                image1 = self.target_transform(image1)

            mask = torch.ones((1, height, width))

            # all_cam_items['camera_' + str(i)] = (image0, image1, mask, mask, rot_image1,
            # rot_image2, rot_image3, img_path0, img_path1)

            all_cam_items['image'].append((image0, image1))
            all_cam_items['mask'].append((mask, mask))
            all_cam_items['rotation'].append((rot_image1, rot_image2, rot_image3))
            # all_cam_items['image_path'].append((img_path0, img_path1))

        # Process calibration

        # The calibration ordering here have to be the same as looping through all cameras above
        all_cam_items['calib_intrinsics'] = intrinsics
        all_cam_items['calib_extrinsics'] = calib['extrinsics']
        all_cam_items['calib_distortions'] = calib['distortions']
        all_cam_items['calib_names'] = calib['names']

        return all_cam_items


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


if __name__ == '__main__':
    import torchvision.transforms as transforms
    root = '~/data/anipose/release/fly-testing'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = FlyDataset(root,
                         transforms.Compose([
                             transforms.ToTensor(),
                             normalize,]),
                         target_transform=transforms.Compose([
                             transforms.ToTensor(),
                             normalize,]),
                         image_size=[256, 256])

from __future__ import print_function, absolute_import

import os
import os.path

import numpy as np

import torch
import torch.utils.data as data
import torchvision

import torchvision.transforms.functional as TF

from dataloader.data_utils import *

import cv2

import toml
import scipy.io
from glob import glob
from collections import defaultdict
import re

from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

def natural_keys(text):
    return [  int(c) if c.isdigit() else c
              for c in re.split('(\d+)', text) ]

def true_basename(fname):
    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    return basename

def get_cam_name(cam_regex, fname):
    basename = true_basename(fname)
    match = re.search(cam_regex, basename)

    if not match:
        return None
    else:
        name = match.groups()[0]
        return name.strip()

def get_video_name(cam_regex, fname):
    basename = true_basename(fname)
    vidname = re.sub(cam_regex, '', basename)
    return vidname.strip()


def generate_pair_images(root, subjects, actions, gap=20):
    images = []

    root = os.path.expanduser(root)

    # root = /home/ubuntu/efs/video_datasets/rat7m/

    # calibration_root = '/home/ubuntu/efs/video_datasets/rat7m/cameras/'
    calibration_root = os.path.join(root, 'labels')

    cameras = ['camera1','camera2', 'camera3', 'camera4', 'camera5', 'camera6']

    for subject in ['s1-d1']:
        subject_root = os.path.join(root, subject)

        calib_file = os.path.join(calibration_root, "mocap-" + subject + '.mat')

        mat = scipy.io.loadmat(calib_file)

        # ('frame', 'O'), ('IntrinsicMatrix', 'O'), ('rotationMatrix', 'O'), ('translationVector', 'O'), ('TangentialDistortion', 'O'), ('RadialDistortion', 'O')]
        # print(mat['cameras'].item()[0]['IntrinsicMatrix'])

        all_subdir = os.listdir(subject_root)
        all_clips = []
        for subdir in all_subdir:
            if not subdir.endswith('.mp4'):
                all_clips.append(int(subdir.split('-')[-1]))

        all_clips = sorted(set(all_clips))

        for clip in all_clips:

            length = len(os.listdir(os.path.join(subject_root, subject + '-camera1-' + str(clip))))

            for framenum in range(0, length - gap, 30):

                allcams_item = []

                for cam in cameras:

                    clip_root = os.path.join(subject_root, subject + '-' + cam +  '-' + str(clip))

                    name_0 = 'frame{:08d}.jpg'.format(framenum)
                    name_1 = 'frame{:08d}.jpg'.format(framenum+gap)
                    im0 = os.path.join(clip_root, name_0)
                    im1 = os.path.join(clip_root, name_1)
                    bbox0 = [0,0,0,0]
                    bbox1 = [0,0,0,0]
                    item = [im0, im1, bbox0, bbox1]
                    allcams_item.append(item)

                images.append({"calibration": mat,
                               "items": allcams_item})


    return images


def generate_pair_images_videos(root, subjects, gap=20, downsample=480):
    root = os.path.abspath(os.path.expanduser(root))
    subjects = ['s1-d1']

    camera_names = ['camera1','camera2', 'camera3', 'camera4', 'camera5', 'camera6']
    calibration_root = os.path.join(root, 'labels')

    cam_regex = '(camera[0-9]+)-'

    outs = []
    for subject in subjects:
        subject_root = os.path.join(root, subject)
        calib_file = os.path.join(calibration_root, "mocap-" + subject + '.mat')
        calibration = load_calibration_rat7m(calib_file)

        video_files = glob(os.path.join(root, subject, '*.mp4'))
        cam_videos = defaultdict(list)
        for vf in video_files:
            name = get_video_name(cam_regex, vf)
            cam_videos[name].append(vf)

        vid_names = cam_videos.keys()
        vid_names = sorted(vid_names, key=natural_keys)[:5]

        for name in tqdm(vid_names, ncols=70, desc=subject):
            fnames = cam_videos[name]
            cam_names = [get_cam_name(cam_regex, f) for f in fnames]
            fname_dict = dict(zip(cam_names, fnames))

            calib_temp = {'names': [],
                          'intrinsics': [],
                          'extrinsics': [],
                          'distortions': []}

            video_images = []
            video_images_next = []
            for ix_calib, cname in enumerate(camera_names):
                if cname in fname_dict:
                    calib_temp['names'].append(cname)
                    calib_temp['intrinsics'].append(calibration['intrinsics'][ix_calib])
                    calib_temp['extrinsics'].append(calibration['extrinsics'][ix_calib])
                    calib_temp['distortions'].append(calibration['distortions'][ix_calib])

                    images = []
                    images_next = []

                    cap = cv2.VideoCapture(fname_dict[cname])
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for framenum in range(0, frame_count, downsample):
                        c = cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * framenum / fps)
                        ret, img_bgr = cap.read()
                        if not ret: break
                        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        images.append(img)

                    for framenum in range(gap, frame_count, downsample):
                        c = cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * framenum / fps)
                        ret, img_bgr = cap.read()
                        if not ret: break
                        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        images_next.append(img)
                    cap.release()

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
    # rotmat, _ = cv2.Rodrigues(np.array(rvec))
    out[:3,:3] = rvec
    out[:3, 3] = np.array(tvec).flatten()
    out[3, 3] = 1
    return out


def load_calibration_rat7m(calib_file):
    mat = scipy.io.loadmat(calib_file)

    intrinsics = []
    extrinsics = []
    distortions = []

    names = [x.lower() for x in mat['cameras'].dtype.names]

    for cam in [0, 1, 4, 2, 3, 5]:
        intrinsics.append(mat['cameras'].item()[cam]['IntrinsicMatrix'].item().transpose())

        # print(mat['cameras'].item()[cam]['rotationMatrix'].shape)
        extrinsics.append(make_M(mat['cameras'].item()[cam]['rotationMatrix'].item().transpose(),
                        mat['cameras'].item()[cam]['translationVector'].item()))

        # print(mat['cameras'].item()[cam]['RadialDistortion'].item())
        distortions.append([mat['cameras'].item()[cam]['RadialDistortion'].item()[0,0],
            mat['cameras'].item()[cam]['RadialDistortion'].item()[0,1],
            mat['cameras'].item()[cam]['TangentialDistortion'].item()[0,0],
            mat['cameras'].item()[cam]['TangentialDistortion'].item()[0,1], 0.0])

    return {
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'distortions': distortions
    }



class RatDataset(data.Dataset):
    """DataLoader for Human 3.6M dataset
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=box_loader, image_size=[128, 128],
                 simplified=False, crop_box=True, frame_gap=20):

        subjects = ['s1-d1']
        actions = []

        #  'S5','S6', 'S7',

        # if simplified:
        #     actions = ['Waiting-1', 'Waiting-2',
        #         'Posing-1', 'Posing-2', 'Greeting-1', 'Greeting-2',
        #         'Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2', 'Walking-1', 'Walking-2']
        # else:
        #     actions = ['Directions-1', 'Eating-1', 'Phoning-1', 'Purchases-1',
        #                'SittingDown-1', 'TakingPhoto-1', 'Walking-1', 'WalkingTogether-1',
        #                'Directions-2', 'Eating-2', 'Phoning-2', 'Purchases-2',
        #                'SittingDown-2', 'TakingPhoto-2', 'Walking-2', 'WalkingTogether-2',
        #                'Discussion-1', 'Greeting-1', 'Posing-1', 'Sitting-1', 'Smoking-1',
        #                'Waiting-1', 'WalkingDog-1', 'Discussion-2', 'Greeting-2', 'Posing-2',
        #                'Sitting-2', 'Smoking-2', 'Waiting-2', 'WalkingDog-2']

            # actions = ['Directions-1']

        samples = generate_pair_images_videos(root, subjects, gap=frame_gap)

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
        intrinsics = torch.as_tensor(calib['intrinsics']).clone()

        for i in range(num_cameras):
            im0_arr, im1_arr = image_dict['items'][i]

            image0 = Image.fromarray(im0_arr)
            image1 = Image.fromarray(im1_arr)

            height, width = self._image_size[:2]

            ratio = min(image0.height / height, image0.width / width)
            crop_size = (int(ratio * height), int(ratio * width))
            crop_params = transforms.RandomCrop.get_params(image0, crop_size)

            # adjust intrinsics for crop
            # intrinsics[i, :2, 0] -= crop_params[1] # x
            # intrinsics[i, :2, 1] -= crop_params[0] # y
            intrinsics[i, 0, 2] -= crop_params[1] # x
            intrinsics[i, 1, 2] -= crop_params[0] # y

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
        all_cam_items['calib_extrinsics'] = torch.as_tensor(calib['extrinsics'])
        all_cam_items['calib_distortions'] = torch.as_tensor(calib['distortions'])
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

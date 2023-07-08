#!/usr/bin/env ipython3

import numpy as np
import cv2
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import torchvision

from model.unsupervised_model import Model as orgModel
# from model.kpt_detector import Model
from utils.visualize import visualize_with_circles

from dataloader.data_utils import default_loader

from tqdm import tqdm, trange
# from PIL import Image
# import seaborn as sns

import h5py

import toml

def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(np.array(rvec))
    out[:3,:3] = rotmat
    out[:3, 3] = np.array(tvec).flatten()
    out[3, 3] = 1
    return out


# resume, checkpoint, num keypoints
def load_model(resume, output_dir, image_size=256, num_keypoints = 10):

    # model = Model(num_keypoints)

    # # Assume GPU 0
    torch.cuda.set_device(0)
    # model.cuda(0)

    save_dir = os.path.join(output_dir, 'keypoints_confidence')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.mkdir(os.path.join(save_dir, 'train'))
        os.mkdir(os.path.join(save_dir, 'test'))

    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(0)
    checkpoint = torch.load(resume, map_location=loc)

    output_shape = (int(image_size/4), int(image_size/4))
    org_model = orgModel(num_keypoints, output_shape=output_shape)

    org_model.load_state_dict(checkpoint['state_dict'], strict = False)
    org_model_dict = org_model.state_dict()

    # org_model.cuda(0)

    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in org_model_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))

    org_model.eval()

    return org_model
    # model.eval()

    # return model, save_dir

def load_calibration(calib_fname):
    calib = toml.load(calib_fname)
    items = sorted(calib.items())
    cam_names = [d['name'] for c, d in items]
    intrinsics = [d['matrix'] for c, d in items]
    distortions = [d['distortions'] for c, d in items]
    extrinsics = [make_M(d['rotation'], d['translation']) for c, d in items]
    intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32).cuda()
    extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32).cuda()
    distortions = torch.as_tensor(distortions, dtype=torch.float32).cuda()
    return {
        'names': cam_names,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'distortions': distortions
    }

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

def triangulate_batch(points, camera_mats, scores=None):
    """Given a CxNx2 array of points and Cx4x4 array of extrinsics,
    this returns a Nx3 array of points,
    where N is the number of points and C is the number of cameras.
    Optionally also takes a CxN array of scores for points."""
    # we build an A matrix that is (num_cams, num_points, 2d, 4)
    num_cams = len(camera_mats)
    A_base = camera_mats[:, None, 2:3].broadcast_to(num_cams, 1, 2, 4)
    A = points[:, :, :,None] * A_base - camera_mats[:, None, :2]
    if scores is not None:
        A = A * scores[:, :, None, None]
    # now shape A matrix to (num_points, num_cams*2, 4)
    A = A.swapaxes(0, 1).reshape(-1, num_cams*2, 4)
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    p3d = vh[:, -1]
    p3d = p3d[:, :3] / p3d[:,3:4]
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

def triangulate_points_full(points_2d, camera_params, confidence=None):
    extrinsics = camera_params['extrinsics']
    intrinsics = camera_params['intrinsics']
    distortions = camera_params['distortions']
    pts_2d_und = torch.stack(
        [undistort_torch(points_2d[i], intrinsics[i], distortions[i])
         for i in range(len(intrinsics))])
    # pts_3d = torch.stack([triangulate_simple(pts_2d_und[:, i], extrinsics, confidence[:, i])
    #                       for i in range(pts_2d_und.shape[1])])
    pts_3d = triangulate_batch(pts_2d_und, extrinsics, confidence)
    return pts_3d

# load base model
resume = 'checkpoint/H36M/checkpoint.pth.tar'
output_dir = 'output'
image_size = 256
num_keypoints = 32
model = load_model(resume, output_dir, image_size, num_keypoints).cuda()

# for evaluation
valid_joints = (3,2,1,6,7,8,0,12,13,15,27,26,25,17,18,19) + (14,)
bodyparts = ['ankle_r', 'knee_r', 'hip_r',
             'hip_l', 'knee_l', 'ankle_l',
             'pelvis', 'thorax', 'neck_lower', 'head_top',
             'wrist_r', 'elbow_r',
             'shoulder_r', 'shoulder_l',
             'elbow_l', 'wrist_l', 'neck_upper'
]


# sub = 'S9'
# action = 'Eating-1'

train_downsample = 128
test_subs = ['S9', 'S11']

train_p3d_model_l = []
train_p3d_data_l = []
test_p3d_model_l = []
test_p3d_data_l = []

path_h36m = os.path.join('data', 'H36M')
subjects = sorted(os.listdir(path_h36m))

# sub = subjects[0]
sub = 'S9'


print(sub)
actions = sorted(os.listdir(os.path.join(path_h36m, sub)))
# action = actions[0]
action = 'WalkingTogether-1'

prefix = os.path.join("data", "H36M", sub, action)
annot_path = os.path.join(prefix, "annot.h5")


# load calibration
calib_fname = os.path.join("data", "H36M", sub, "calibration.toml")
calib = load_calibration(calib_fname)
cam_names = calib['names']
num_cams = len(cam_names)

# load ground truth data
annos = h5py.File(annot_path, 'r')
camera_nums = annos['camera'][()]
frame_nums = annos['frame'][()]

num_frames = np.max(frame_nums)

cam_frame = dict()
for i in range(len(annos['camera'][()])):
    cam_number = camera_nums[i]
    frame_number = frame_nums[i]
    cam_frame[(cam_number, frame_number)] = i

if sub in test_subs:
    possible_frames = np.arange(1, num_frames, 4)
else:
    possible_frames = np.unique(frame_nums)[::train_downsample]

pts = []
for cname in cam_names:
    cnum = int(cname)
    for inum in possible_frames:
        ix = cam_frame[(cnum, inum)]
        p2d = annos['pose']['2d'][ix][list(valid_joints)]
        pts.append(p2d)
full_points_data = torch.as_tensor(np.array(pts), dtype=torch.float32).cuda()
full_points_data = full_points_data.reshape(num_cams, -1, len(valid_joints), 2)

# poses = np.array(f['pose']['2d'])[:, valid_joints]
# cameras = np.array(f['camera'])
# frames = f['frame'][:]

# load the images
image_paths = [os.path.join(prefix, "imageSequence", cname, 'img_{:06d}.jpg'.format(inum))
               for cname in cam_names
               for inum in possible_frames]

mean_image = [0.485, 0.456, 0.406]
std_image = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean_image,
                                 std=std_image)

trans = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor(),
    normalize])

images = [trans(default_loader(p)) for p in image_paths]


batch_size = 16

# run the model
all_xs = []
all_ys = []
confs = []
for i in trange(0, len(images), batch_size):
    inp = torch.stack(images[i:i+batch_size]).cuda()
    output = model.forward(inp)
    # xs, ys = model.get_keypoints(inp)
    xs, ys = output['pos']
    conf = output['confidence']
    all_xs.append(xs.detach().cpu())
    all_ys.append(ys.detach().cpu())
    confs.append(conf.detach().cpu())
pts = torch.stack([torch.vstack(all_xs), torch.vstack(all_ys)], axis=2)
full_points_model = pts.reshape(len(cam_names), -1, num_keypoints, 2)
full_points_model = (full_points_model + 1) * 500

# triangulate
full_points_data_f = full_points_data.reshape(num_cams, -1, 2).cuda()
full_points_model_f = full_points_model.reshape(num_cams, -1, 2).cuda()

p3d_model_f = triangulate_points_full(full_points_model_f, calib)
p3d_data_f = triangulate_points_full(full_points_data_f, calib)

p3d_model = p3d_model_f.reshape(-1, num_keypoints, 3).cpu().numpy()
p3d_data = p3d_data_f.reshape(-1, len(valid_joints), 3).cpu().numpy()

p2d_model_proj = project_points(p3d_model_f, calib).reshape(full_points_model.shape).cpu().numpy()


images_r = torch.stack(images) \
                .reshape(num_cams, -1, 3, image_size, image_size) \
                .cpu().numpy()

import skvideo.io
outname = os.path.join('output', 'test.avi')
writer = skvideo.io.FFmpegWriter(outname, inputdict={
    # '-hwaccel': 'auto',
    '-framerate': '12',
}, outputdict={
    '-vcodec': 'h264', '-qp': '28',
    '-pix_fmt': 'yuv420p', # to support more players
    '-vf': 'pad=ceil(iw/2)*2:ceil(ih/2)*2' # to handle width/height not divisible by 2
})


for ix_frame in range(images_r.shape[1]):
    out = []
    for ix_cam in range(4):
        im_with_pts = visualize_with_circles(images_r[ix_cam,ix_frame],
                                             np.clip(p2d_model_proj[ix_cam,ix_frame]/500.0, -2, 2),
                                             mean=mean_image, std=std_image)
        im_with_pts = im_with_pts.astype('uint8')
        # im_with_pts = cv2.cvtColor(im_with_pts, cv2.COLOR_RGB2BGR)
        out.append(im_with_pts)
    top = np.hstack(out[:2])
    bot = np.hstack(out[2:])
    s = np.vstack([top, bot])
    writer.writeFrame(s)

# cv2.imwrite(os.path.join('output', 'test.png'), s)
writer.close()

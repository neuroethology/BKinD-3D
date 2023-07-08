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

from model.unsupervised_model_seg_vol_edge_tree import Model

from loss.compute_loss import *
# import visualize

# from utils import Logger, mkdir_p, save_images, save_3d_images, save_multi_images,save_images_2
from utils.model_utils import *
import cv2
import numpy as np
import copy

def load_model(path, model):
    torch.cuda.set_device(0)
    model.cuda(0)    
    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(0)
    checkpoint = torch.load(path, map_location=loc)
    model.load_state_dict(checkpoint['state_dict'], strict = True) 

    return model


def main():

    args = parse_args(create_parser())

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


    
    
    
def main_worker(gpu, ngpus_per_node, args, mode):
    image_size = 256
    nkpts = 15
    # create model
    output_shape = (int(image_size/4), int(image_size/4))
    model = Model(nkpts, output_shape=output_shape)
    model = load_model('/home/amildravid/BKinD-main/2volume_edge/checkpoint_h36m_vf32.pth.tar', model)  
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    
    

    cudnn.benchmark = True

    # Data loading code
    root = os.path.join(args.data)
    

    dataset = load_dataloader(args, mode)
    train_sampler = None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    all_kpts = validate(loader, model, args)
    return all_kpts
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
    # print(points_2d[0])

    extrinsics = camera_params['extrinsics']
    intrinsics = camera_params['intrinsics']
    distortions = camera_params['distortions']
    pts_2d_und = torch.stack(
        [undistort_torch(points_2d[i], intrinsics[i], distortions[i])
         for i in range(len(intrinsics))])

    # print(pts_2d_und[0])

    # pts_3d = torch.stack([triangulate_simple(pts_2d_und[:, i], extrinsics, confidence[:, i])
    #                       for i in range(pts_2d_und.shape[1])])
    pts_3d = triangulate_batch(pts_2d_und, extrinsics, confidence)

    # print(pts_3d)

    return pts_3d


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = np.array(R).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])

#############################



def validate(loader, model, args):
    all_3d_kpts = []
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    edge_weights = torch.randn(105, requires_grad=False, device = torch.device('cuda:'+str(args.gpu)))
    
    with torch.no_grad():
        end = time.time()
        for i, all_cam_items in enumerate(loader):
            all_cam_inputs = []
            all_cam_tr_inputs = []
            all_cams = []
            for cam_num in range(len(all_cam_items['image'])):
                inputs, tr_inputs = all_cam_items['image'][cam_num]
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                tr_inputs = tr_inputs.cuda(args.gpu, non_blocking=True)
                all_cam_inputs.append(inputs)
                all_cam_tr_inputs.append(tr_inputs)

            intrinsics = all_cam_items['calib_intrinsics']
            extrinsics = all_cam_items['calib_extrinsics']
            distortions = all_cam_items['calib_distortions']



            for view in range(extrinsics.size()[1]):
                curr_batch = []
                for batch in range(extrinsics.size()[0]):
                    curr_batch.append(Camera(extrinsics[batch, view, :3, :3], extrinsics[batch, view, :3, 3], 
                        intrinsics[batch, view], distortions[batch, view]))
                all_cams.append(curr_batch)

            output = model(all_cam_inputs, all_cam_tr_inputs, edge_weights, all_cams, all_cam_items)
            all_3d_kpts.append(output['tr_pos_3d'])
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    all_3d_kpts = torch.cat(all_3d_kpts,0)
    
    return all_3d_kpts


if __name__ == '__main__':
    main()

#!/usr/bin/env ipython

import torch
import toml
import cv2
import numpy as np

def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(np.array(rvec))
    out[:3,:3] = rotmat
    out[:3, 3] = np.array(tvec).flatten()
    out[3, 3] = 1
    return out

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

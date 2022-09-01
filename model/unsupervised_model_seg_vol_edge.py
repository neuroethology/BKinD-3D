import torch
import torch.nn as nn

from .resnet_updated import conv3x3
from .resnet_updated import resnetbank50all as resnetbank50
from .globalNet import globalNet

import math

# from model.transformer_model import PerceiverForKeypoints
from model.transformer_model_seg import DetrForKeypoints
import transformers

from copy import deepcopy
import numpy as np

from model.v2v import V2VModel

import torch.nn.functional as F
import itertools

import numpy as np

def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous
    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M
    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean
    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M
    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")



class Point3D:
    def __init__(self, point, size=3, color=(0, 0, 255)):
        self.point = point
        self.size = size
        self.color = color

    def render(self, proj_matrix, canvas):
        point_2d = project_3d_points_to_image_plane_without_distortion(
            proj_matrix, np.array([self.point])
        )[0]

        point_2d = tuple(map(int, point_2d))
        cv2.circle(canvas, point_2d, self.size, self.color, self.size)

        return canvas


class Line3D:
    def __init__(self, start_point, end_point, size=2, color=(0, 0, 255)):
        self.start_point, self.end_point = start_point, end_point
        self.size = size
        self.color = color

    def render(self, proj_matrix, canvas):
        start_point_2d, end_point_2d = project_3d_points_to_image_plane_without_distortion(
            proj_matrix, np.array([self.start_point, self.end_point])
        )

        start_point_2d = tuple(map(int, start_point_2d))
        end_point_2d = tuple(map(int, end_point_2d))

        cv2.line(canvas, start_point_2d, end_point_2d, self.color, self.size)

        return canvas


class Cuboid3D:
    def __init__(self, position, sides):
        self.position = position
        self.sides = sides

    def build(self):
        primitives = []

        line_color = (255, 255, 0)

        start = self.position + np.array([0, 0, 0])
        primitives.append(Line3D(start, start + np.array([self.sides[0], 0, 0]), color=(255, 0, 0)))
        primitives.append(Line3D(start, start + np.array([0, self.sides[1], 0]), color=(0, 255, 0)))
        primitives.append(Line3D(start, start + np.array([0, 0, self.sides[2]]), color=(0, 0, 255)))

        start = self.position + np.array([self.sides[0], 0, self.sides[2]])
        primitives.append(Line3D(start, start + np.array([-self.sides[0], 0, 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, self.sides[1], 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, 0, -self.sides[2]]), color=line_color))

        start = self.position + np.array([self.sides[0], self.sides[1], 0])
        primitives.append(Line3D(start, start + np.array([-self.sides[0], 0, 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, -self.sides[1], 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, 0, self.sides[2]]), color=line_color))

        start = self.position + np.array([0, self.sides[1], self.sides[2]])
        primitives.append(Line3D(start, start + np.array([self.sides[0], 0, 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, -self.sides[1], 0]), color=line_color))
        primitives.append(Line3D(start, start + np.array([0, 0, -self.sides[2]]), color=line_color))

        return primitives

    def render(self, proj_matrix, canvas):
        # TODO: support rotation

        primitives = self.build()

        for primitive in primitives:
            canvas = primitive.render(proj_matrix, canvas)

        return canvas


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        
        if self.upsample is not None:
            x = self.upsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out
    

class Decoder(nn.Module):
    def __init__(self, in_planes=256, wh=14, n_kps=10, ratio=1.0):
        super(Decoder, self).__init__()
        
        self.K = n_kps
        
        w, h = wh, wh
        if ratio != 1.0:
            w = wh
            h = ratio * wh
            
        self.layer1 = BasicBlock(int(in_planes)+self.K, int(in_planes/2),
                                 upsample=nn.Upsample((int(h*2), int(w*2)), mode='bilinear')); in_planes /= 2
        self.layer2 = BasicBlock(int(in_planes)+self.K, int(in_planes/2), 
                                 upsample=nn.Upsample((int(h*4), int(w*4)), mode='bilinear')); in_planes /= 2
        self.layer3 = BasicBlock(int(in_planes)+self.K, int(in_planes/2), 
                                 upsample=nn.Upsample((int(h*8), int(w*8)), mode='bilinear')); in_planes /= 2
        self.layer4 = BasicBlock(int(in_planes)+self.K, int(in_planes/2),
                                 upsample=nn.Upsample((int(h*16), int(w*16)), mode='bilinear')); in_planes /= 2
        self.layer5 = BasicBlock(int(in_planes)+self.K, max(int(in_planes/2), 32),
                                 upsample=nn.Upsample((int(h*32), int(w*32)), mode='bilinear'))
        in_planes = max(int(in_planes/2), 32)
        
        self.conv_final = nn.Conv2d(int(in_planes), 3, kernel_size=1, stride=1)
        
    def forward(self, x, heatmap):
        
        x = torch.cat((x[0], heatmap[0]), dim=1)
        x = self.layer1(x)
        x = torch.cat((x, heatmap[1]), dim=1)
        x = self.layer2(x)
        x = torch.cat((x, heatmap[2]), dim=1)
        x = self.layer3(x)
        x = torch.cat((x, heatmap[3]), dim=1)
        x = self.layer4(x)
        
        x = torch.cat((x, heatmap[4]), dim=1)
        x = self.layer5(x)
        x = self.conv_final(x)
        
        return x
    


def integrate_tensor_3d_with_coordinates(volumes, coord_volumes, softmax=True):
    batch_size, n_volumes, x_size, y_size, z_size = volumes.shape

    volumes = volumes.reshape((batch_size, n_volumes, -1))
    if softmax:
        volumes = nn.functional.softmax(volumes, dim=2)
    else:
        volumes = nn.functional.relu(volumes)

    volumes = volumes.reshape((batch_size, n_volumes, x_size, y_size, z_size))
    coordinates = torch.einsum("bnxyz, bxyzc -> bnc", volumes, coord_volumes)

    return coordinates, volumes

    

def unproject_heatmaps(heatmaps, proj_matricies, coord_volumes, volume_aggregation_method='sum', vol_confidences=None):
    device = heatmaps.device
    batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])
    volume_shape = coord_volumes.shape[1:4]

    volume_batch = torch.zeros(batch_size, n_joints, *volume_shape, device=device)

    # TODO: speed up this this loop
    for batch_i in range(batch_size):
        coord_volume = coord_volumes[batch_i]
        grid_coord = coord_volume.reshape((-1, 3))

        volume_batch_to_aggregate = torch.zeros(n_views, n_joints, *volume_shape, device=device)

        for view_i in range(n_views):
            heatmap = heatmaps[batch_i, view_i]
            heatmap = heatmap.unsqueeze(0)

            grid_coord_proj = project_3d_points_to_image_plane_without_distortion(
                proj_matricies[batch_i, view_i], grid_coord, convert_back_to_euclidean=False
            )

            invalid_mask = grid_coord_proj[:, 2] <= 0.0  # depth must be larger than 0.0

            grid_coord_proj[grid_coord_proj[:, 2] == 0.0, 2] = 1.0  # not to divide by zero
            grid_coord_proj = homogeneous_to_euclidean(grid_coord_proj)

            # transform to [-1.0, 1.0] range
            grid_coord_proj_transformed = torch.zeros_like(grid_coord_proj)
            grid_coord_proj_transformed[:, 0] = 2 * (grid_coord_proj[:, 0] / heatmap_shape[0] - 0.5)
            grid_coord_proj_transformed[:, 1] = 2 * (grid_coord_proj[:, 1] / heatmap_shape[1] - 0.5)
            grid_coord_proj = grid_coord_proj_transformed

            # prepare to F.grid_sample
            grid_coord_proj = grid_coord_proj.unsqueeze(1).unsqueeze(0)
            try:
                current_volume = F.grid_sample(heatmap, grid_coord_proj, align_corners=True)
            except TypeError: # old PyTorch
                current_volume = F.grid_sample(heatmap, grid_coord_proj)

            # zero out non-valid points
            current_volume = current_volume.view(n_joints, -1)
            current_volume[:, invalid_mask] = 0.0

            # reshape back to volume
            current_volume = current_volume.view(n_joints, *volume_shape)

            # collect
            volume_batch_to_aggregate[view_i] = current_volume

        # agregate resulting volume
        if volume_aggregation_method.startswith('conf'):
            volume_batch[batch_i] = (volume_batch_to_aggregate * vol_confidences[batch_i].view(n_views, n_joints, 1, 1, 1)).sum(0)
        elif volume_aggregation_method == 'sum':
            volume_batch[batch_i] = volume_batch_to_aggregate.sum(0)
        elif volume_aggregation_method == 'max':
            volume_batch[batch_i] = volume_batch_to_aggregate.max(0)[0]
        elif volume_aggregation_method == 'softmax':
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate.clone()
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, -1)
            volume_batch_to_aggregate_softmin = nn.functional.softmax(volume_batch_to_aggregate_softmin, dim=0)
            volume_batch_to_aggregate_softmin = volume_batch_to_aggregate_softmin.view(n_views, n_joints, *volume_shape)

            volume_batch[batch_i] = (volume_batch_to_aggregate * volume_batch_to_aggregate_softmin).sum(0)
        else:
            raise ValueError("Unknown volume_aggregation_method: {}".format(volume_aggregation_method))

    return volume_batch


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


class Model(nn.Module):
    def __init__(self, n_kps=10, output_dim=200, pretrained=True, output_shape=(64, 64)):
        
        super(Model, self).__init__()
        self.K = n_kps
        
        channel_settings = [2048, 1024, 512, 256]
        self.output_shape = output_shape

        self.kptNet = globalNet(channel_settings, output_shape, n_kps)

        self.volume_size = 64
        self.cuboid_side = 12500
        
        self.ch_softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.volume_net = V2VModel(128, self.K)

        # width == height for now
        self.decoder = Decoder(in_planes=2048, wh=int(output_shape[0]/8), n_kps=2, ratio=1.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.encoder = resnetbank50(pretrained=pretrained)

        self.softplus = nn.Softplus()

    def get_keypoints(self, x):
        x_res = self.encoder(x)

        # Get keypoints of x
        kpt_feat, kpt_out = self.kptNet(x_res)  # keypoint for reconstruction
        
        # Reconstruction module
        heatmap = kpt_out[-1].view(-1, self.K, kpt_out[-1].size(2) * kpt_out[-1].size(3))
        heatmap = self.ch_softmax(heatmap)
        heatmap = heatmap.view(-1, self.K, kpt_out[-1].size(2), kpt_out[-1].size(3))
                
        u_x, u_y, covs = self._mapTokpt(heatmap)        

        return (u_x, u_y)        

    def get_all_keypoints(self, x_list):

        u_x_list = []
        u_y_list = []
        for item in x_list:
            u_x, u_y = self.get_keypoints(item)
            u_x_list.append(u_x)
            u_y_list.append(u_y)
            

        return (u_x_list, u_y_list)


    def forward(self, x, tr_x, edge_weights, all_cams, all_cam_items): #gmtr_x1 = None, gmtr_x2 = None, gmtr_x3 = None):
        
        # x and tr_x are lists 
        device = x[0].device

        return_dict = {}

        heatmaps_ori = []
        heatmaps_tr = []

        heatmap_list = []

        x_res_list = []

        pt_2d = []
        pt_2d_tr = []

        mask = torch.cuda.FloatTensor(x[0].size()).uniform_() > 0.8

        for index in range(len(x)):

            x_res = self.encoder(x[index])
            tr_x_res = self.encoder(tr_x[index])


            # masked_x_res = self.encoder(x[index]*mask)

            x_res_list.append(x_res)


            tr_kpt_feat, tr_kpt_out = self.kptNet(tr_x_res)  # keypoint for reconstruction

            tr_heatmap = tr_kpt_out[-1]

            tr_confidence = tr_heatmap.max(dim=-1)[0].max(dim=-1)[0]

            heatmaps_tr.append(tr_heatmap)
            heatmap_list.append(tr_heatmap)
            # Get keypoints of x


            kpt_feat, kpt_out = self.kptNet(x_res)  # keypoint for reconstruction
            
            heatmap = kpt_out[-1]
                    
            confidence = heatmap.max(dim=-1)[0].max(dim=-1)[0]

            heatmaps_ori.append(heatmap)

        
        new_cameras = deepcopy(all_cams)
        n_views = len(x)
        batch_size = x[0].size()[0]
        #### Get projection matrices
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize([1000, 1000], [kpt_out[-1].size(2), kpt_out[-1].size(3)])

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)        


        # print(proj_matricies.size())
        # error

        heatmaps_tr = torch.stack(heatmaps_tr, dim = 1)


        heatmaps_ori = torch.stack(heatmaps_ori, dim = 1)

        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):

            base_point = torch.from_numpy(np.array([0,0,0]))#.to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = Cuboid3D(position, sides)

            # cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)


            coord_volumes[batch_i] = coord_volume


        # unproject original
        volume = unproject_heatmaps(heatmaps_ori, proj_matricies, coord_volumes, volume_aggregation_method = 'softmax')

        # integral 3d
        volume = self.volume_net(volume)
        vol_keypoints_3d, volume = integrate_tensor_3d_with_coordinates(volume, coord_volumes, softmax=True)


        ####################
        # unproject original
        volume_tr = unproject_heatmaps(heatmaps_tr, proj_matricies, coord_volumes, volume_aggregation_method = 'softmax')

        # integral 3d
        volume_tr = self.volume_net(volume_tr)
        vol_keypoints_3d_tr, volume_tr = integrate_tensor_3d_with_coordinates(volume_tr, coord_volumes, softmax=True)


        intrinsics = all_cam_items['calib_intrinsics'].to(device)
        extrinsics = all_cam_items['calib_extrinsics'].to(device)
        distortions = all_cam_items['calib_distortions'].to(device)


        #torch.Size([2, 15, 3])
        keypoint_batch = []

        keypoint_batch_tr = []

        for b in range(batch_size):

            camera_params = {
                'extrinsics': extrinsics[b],
                'intrinsics': intrinsics[b],
                'distortions': distortions[b]
            }

            keypoint_2d = project_points(vol_keypoints_3d[b], camera_params)/500 - 1

            keypoint_batch.append(keypoint_2d)

            keypoint_2d_tr = project_points(vol_keypoints_3d_tr[b], camera_params)/500 - 1

            keypoint_batch_tr.append(keypoint_2d_tr)


        keypoint_batch = torch.stack(keypoint_batch, dim = 0)
        keypoint_batch_tr = torch.stack(keypoint_batch_tr, dim = 0)

        recons = []

        tr_pos_cam = []
        pos_cam = []

        pt_2d_recon = []

        tr_kp_cond_list =[]

        #########################################
        # Decoding        
        for index in range(len(x)):

            u_x = keypoint_batch[:, index, :, 0].clip(-1, 1)
            u_y = keypoint_batch[:, index, :, 1].clip(-1, 1)

            tr_u_x = keypoint_batch_tr[:, index, :, 0].clip(-1, 1)
            tr_u_y = keypoint_batch_tr[:, index, :, 1].clip(-1, 1)

            # print(u_x, u_y)

            tr_pos_cam.append((tr_u_x, tr_u_y))
            pos_cam.append((u_x, u_y))

            tr_kpt_conds = []
            
            prev_w, prev_h = int(self.output_shape[0]/16), int(self.output_shape[1]/16)
            std_in = [0.1, 0.1, 0.01, 0.01, 0.001]

            
            for i in range(0, 5):
                prev_h *= 2;  prev_w *= 2
                
                # All pairs!
                hmaps = self._kptToEdge(u_x, u_y, u_x, u_y, H=prev_h, W=prev_w, inv_std=std_in[i], normalize=False) 

                # 1 x 105 x 4 x 4

                weighted = torch.einsum('mbch,b->mbch', hmaps, self.softplus(edge_weights))


                hmaps, _ = torch.max(weighted, dim = 1, keepdim = True)

                hmaps_2 = self._kptToEdge(tr_u_x, tr_u_y, tr_u_x, tr_u_y, H=prev_h, W=prev_w, inv_std=std_in[i], normalize=False) 

                weighted = torch.einsum('mbch,b->mbch', hmaps_2, self.softplus(edge_weights))


                hmaps_2, _ = torch.max(weighted, dim = 1, keepdim = True)


                hmaps = torch.cat([hmaps, hmaps_2], dim = 1)

                tr_kpt_conds.append(hmaps)
            
            recon = self.decoder(x_res_list[index], tr_kpt_conds)

            recons.append(recon)

            tr_kp_cond_list.append(tr_kpt_conds[-1])


        return_dict['recon'] = recons

        # return_dict['recon_2d'] = pt_2d_recon

        return_dict['tr_pos'] = tr_pos_cam
        return_dict['pos'] = pos_cam
        return_dict['confidence'] = torch.ones(u_x.size())
        return_dict['tr_heatmap'] = tr_kpt_conds[-1]
        return_dict['tr_kpt_out'] = heatmap_list
        return_dict['tr_confidence'] = torch.ones(u_x.size()) #tr_confidence

        return_dict['tr_pos_3d'] = vol_keypoints_3d_tr
        return_dict['pos_3d'] = vol_keypoints_3d


        return_dict['tr_kpt_cond'] = tr_kp_cond_list
        # return_dict['kp_cond'] = kp_cond_list

        ################# Potentially add 2D recon loss back?


        # return_dict['logits'] = output.logits
        # return_dict['tr_logits'] = tr_output.logits


        # if gmtr_x1 is not None:  # Rotation loss
        #     out_h, out_w = int(self.output_shape[0]*2), int(self.output_shape[1]*2)
            
        #     # gmtr_x_res = self.encoder(gmtr_x1)
        #     # gmtr_kpt_feat, gmtr_kpt_out = self.kptNet(gmtr_x_res)

        #     gmtr1_output, _, _ = self.kptNet(gmtr_x1)

        #     gmtr1_kpt_out = gmtr1_output.pred_masks
        #     gmtr_heatmap = gmtr1_kpt_out.view(-1, self.K, gmtr1_kpt_out.size(2) * gmtr1_kpt_out.size(3))
        #     gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
        #     gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr1_kpt_out.size(2), gmtr1_kpt_out.size(3))
            
        #     gmtr_u_x, gmtr_u_y, gmtr_covs = self._mapTokpt(gmtr_heatmap)

        #     gmtr_kpt_conds_1 = self._kptTomap(gmtr_u_x, gmtr_u_y, H=out_h, W=out_w, inv_std=0.001, normalize=False)

        #     #################################################
        #     gmtr2_output, _, _ = self.kptNet(gmtr_x2)
            
        #     gmtr2_kpt_out = gmtr2_output.pred_masks            
        #     gmtr_heatmap = gmtr2_kpt_out.view(-1, self.K, gmtr2_kpt_out.size(2) * gmtr2_kpt_out.size(3))
        #     gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
        #     gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr2_kpt_out.size(2), gmtr2_kpt_out.size(3))
            
        #     gmtr_u_x_2, gmtr_u_y_2, gmtr_covs = self._mapTokpt(gmtr_heatmap)

        #     gmtr_kpt_conds_2 = self._kptTomap(gmtr_u_x_2, gmtr_u_y_2, H=out_h, W=out_w, inv_std=0.001, normalize=False)

        #     ###########################################
        #     gmtr3_output, _, _ = self.kptNet(gmtr_x3)
            
        #     gmtr3_kpt_out = gmtr3_output.pred_masks                        
        #     gmtr_heatmap = gmtr3_kpt_out.view(-1, self.K, gmtr3_kpt_out.size(2) * gmtr3_kpt_out.size(3))
        #     gmtr_heatmap = self.ch_softmax(gmtr_heatmap)
        #     gmtr_heatmap = gmtr_heatmap.view(-1, self.K, gmtr3_kpt_out.size(2), gmtr3_kpt_out.size(3))
            
        #     gmtr_u_x_3, gmtr_u_y_3, gmtr_covs = self._mapTokpt(gmtr_heatmap)

        #     gmtr_kpt_conds_3 = self._kptTomap(gmtr_u_x_3, gmtr_u_y_3, H=out_h, W=out_w, inv_std=0.001, normalize=False)

        #     return_dict['gmtr_pos'] = (gmtr_u_x, gmtr_u_y, gmtr_u_x_2, gmtr_u_y_2, gmtr_u_x_3, gmtr_u_y_3)
        #     return_dict['gmtr_heatmap'] = (gmtr_kpt_conds_1, gmtr_kpt_conds_2, gmtr_kpt_conds_3)
        #     return_dict['gmtr_kpt_out'] = gmtr1_kpt_out[-1]  
        
        return return_dict
    

    def cross_view_recon(self, x, u_x, u_y, tr_u_x, tr_u_y):
        
        return_dict = {}

        output, x_res, mask_1 = self.kptNet(x)
        
        tr_kpt_conds = []
        
        prev_w, prev_h = int(self.output_shape[0]/16), int(self.output_shape[1]/16)
        std_in = [0.1, 0.1, 0.01, 0.01, 0.001]
        
        for i in range(0, 5):
            prev_h *= 2;  prev_w *= 2
            
            # _We can concatenate both keypoint representation
            hmaps = self._kptTomap(tr_u_x, tr_u_y, H=prev_h, W=prev_w, inv_std=std_in[i], normalize=False)

            hmaps_2 = self._kptTomap(u_x, u_y, H=prev_h, W=prev_w, inv_std=std_in[i], normalize=False)

            hmaps = torch.cat([hmaps, hmaps_2], dim = 1)

            tr_kpt_conds.append(hmaps)
            
        recon = self.decoder(x_res, tr_kpt_conds)

        return_dict['recon'] = recon

 
        return return_dict

        
    def _mapTokpt(self, heatmap):
        # heatmap: (N, K, H, W)    
            
        H = heatmap.size(2)
        W = heatmap.size(3)
        
        s_y = heatmap.sum(3)  # (N, K, H)
        s_x = heatmap.sum(2)  # (N, K, W)
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        
        u_y = (y * s_y).sum(2) / s_y.sum(2)  # (N, K)
        u_x = (x * s_x).sum(2) / s_x.sum(2)
        
        y = torch.reshape(y, (1, 1, H, 1))
        x = torch.reshape(x, (1, 1, 1, W))
        
        # Covariance
        var_y = ((heatmap * y.pow(2)).sum(2).sum(2) - u_y.pow(2)).clamp(min=1e-6)
        var_x = ((heatmap * x.pow(2)).sum(2).sum(2) - u_x.pow(2)).clamp(min=1e-6)
        
        cov = ((heatmap * (x - u_x.view(-1, self.K, 1, 1)) * (y - u_y.view(-1, self.K, 1, 1))).sum(2).sum(2)) #.clamp(min=1e-6)
                
        return u_x, u_y, (var_x, var_y, cov)
    
    
    def _kptTomap(self, u_x, u_y, inv_std=1.0/0.1, H=16, W=16, normalize=False):        
        mu_x = u_x.unsqueeze(2).unsqueeze(3)  # (N, K, 1, 1)
        mu_y = u_y.unsqueeze(2) 
        
        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        y = torch.reshape(y, (1, H))
        x = torch.reshape(x, (1, 1, W))
        
        g_y = (mu_y - y).pow(2)
        g_x = (mu_x - x).pow(2)
        
        g_y = g_y.unsqueeze(3)
        g_yx = g_y + g_x
        
        g_yx = torch.exp(- g_yx / (2 * inv_std) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        
        if normalize:
            g_yx = g_yx / g_yx.sum(2, keepdim=True).sum(3, keepdim=True)
        
        return g_yx


    def _kptToEdge(self, u_x, u_y, v_x, v_y, H=16, W=16, inv_std=1.0/0.1, normalize=False):
        

        pairs = np.array(list(itertools.combinations(range(15), 2)))

        y = torch.linspace(-1.0, 1.0, H).cuda()
        x = torch.linspace(-1.0, 1.0, W).cuda()
        y = torch.reshape(y, (1, 1, H, 1, 1))
        x = torch.reshape(x, (1, 1, 1, W, 1))
        
        mu_u_x = u_x.unsqueeze(2)  # (N, K, 1)
        mu_u_y = u_y.unsqueeze(2)  # (N, K, 1)
        
        mu_v_x = v_x.unsqueeze(2)
        mu_v_y = v_y.unsqueeze(2)
        
        mu_u_x = mu_u_x[:,pairs[:,0],:]
        mu_u_y = mu_u_y[:,pairs[:,0],:]
                
        mu_v_x = mu_v_x[:,pairs[:,1],:]
        mu_v_y = mu_v_y[:,pairs[:,1],:]
        
        alpha = torch.linspace(0, 1.0, H).cuda()
        eq_x = alpha * mu_u_x + (1 - alpha) * mu_v_x  # Representative points (N, K, #points)
        eq_y = alpha * mu_u_y + (1 - alpha) * mu_v_y
        eq_x = eq_x.unsqueeze(2).unsqueeze(3)
        eq_y = eq_y.unsqueeze(2).unsqueeze(3)
        
        min_dist = ((x - eq_x).pow(2) + (y - eq_y).pow(2)) #.sqrt()  # (N, K, H, W, #pts)
        min_dist, _ = torch.min(min_dist, dim=4)
        
        g_yx = torch.exp( - min_dist / (2 * inv_std) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        # g_yx = torch.exp(- ((x - eq_x).pow(2) + (y - eq_y).pow(2)) / (2 * inv_std) ) * 1 / math.sqrt(2 * math.pi * inv_std)
        
        if normalize:
            g_yx = g_yx / g_yx.sum(2, keepdim=True).sum(3, keepdim=True)
            
        return g_yx

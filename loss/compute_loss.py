## Sum all losses here
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.separation_loss import separation_loss
from loss.rotation_loss import rotation_loss
from loss.ssim_loss import compute_ssim

import numpy as np
import itertools

from model.vggforLoss import mse_loss_mask, VGG_feat

from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union, Tuple

import warnings
# threshold for checking that point crosscorelation
# is full rank in corresponding_points_alignment
AMBIGUOUS_ROT_SINGULAR_THR = 1e-15


# named tuples for inputs/outputs
class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor


def wmean(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: Union[int, Tuple[int]] = -2,
    keepdim: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Finds the mean of the input tensor across the specified dimension.
    If the `weight` argument is provided, computes weighted mean.
    Args:
        x: tensor of shape `(*, D)`, where D is assumed to be spatial;
        weights: if given, non-negative tensor of shape `(*,)`. It must be
            broadcastable to `x.shape[:-1]`. Note that the weights for
            the last (spatial) dimension are assumed same;
        dim: dimension(s) in `x` to average over;
        keepdim: tells whether to keep the resulting singleton dimension.
        eps: minimum clamping value in the denominator.
    Returns:
        the mean tensor:
        * if `weights` is None => `mean(x, dim)`,
        * otherwise => `sum(x*w, dim) / max{sum(w, dim), eps}`.
    """
    args = {"dim": dim, "keepdim": keepdim}

    if weight is None:
        # pyre-fixme[6]: For 1st param expected `Optional[dtype]` but got
        #  `Union[Tuple[int], int]`.
        return x.mean(**args)

    if any(
        xd != wd and xd != 1 and wd != 1
        for xd, wd in zip(x.shape[-2::-1], weight.shape[::-1])
    ):
        raise ValueError("wmean: weights are not compatible with the tensor")

    # pyre-fixme[6]: For 1st param expected `Optional[dtype]` but got
    #  `Union[Tuple[int], int]`.
    return (x * weight[..., None]).sum(**args) / weight[..., None].sum(**args).clamp(
        eps
    )


def corresponding_points_alignment(
    X: Union[torch.Tensor, "Pointclouds"],
    Y: Union[torch.Tensor, "Pointclouds"],
    weights: Union[torch.Tensor, List[torch.Tensor], None] = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9,
) -> SimilarityTransform:
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:
    `s[i] X[i] R[i] + T[i] = Y[i]`,
    for all batch indexes `i` in the least squares sense.
    The algorithm is also known as Umeyama [1].
    Args:
        **X**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **weights**: Batch of non-negative weights of
            shape `(minibatch, num_point)` or list of `minibatch` 1-dimensional
            tensors that may have different shapes; in that case, the length of
            i-th tensor should be equal to the number of points in X_i and Y_i.
            Passing `None` means uniform weights.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **eps**: A scalar for clamping to avoid dividing by zero. Active for the
            code that estimates the output scale `s`.
    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.
    References:
        [1] Shinji Umeyama: Least-Suqares Estimation of
        Transformation Parameters Between Two Point Patterns
    """

    # make sure we convert input Pointclouds structures to tensors
    # Xt, num_points = oputil.convert_pointclouds_to_tensor(X)
    # Yt, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)

    Xt = X
    Yt = Y

    num_points = X.shape[1] * torch.ones(  # type: ignore
            # pyre-fixme[16]: Item `Pointclouds` of `Union[Pointclouds, Tensor]` has
            #  no attribute `shape`.
            X.shape[0],
            device=X.device,
            dtype=torch.int64,
        )

    num_points_Y = Y.shape[1] * torch.ones(  # type: ignore
            # pyre-fixme[16]: Item `Pointclouds` of `Union[Pointclouds, Tensor]` has
            #  no attribute `shape`.
            Y.shape[0],
            device=Y.device,
            dtype=torch.int64,
        )

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )
    if weights is not None:
        if isinstance(weights, list):
            if any(np != w.shape[0] for np, w in zip(num_points, weights)):
                raise ValueError(
                    "number of weights should equal to the "
                    + "number of points in the point cloud."
                )
            weights = [w[..., None] for w in weights]
            weights = strutil.list_to_padded(weights)[..., 0]

        if Xt.shape[:2] != weights.shape:
            raise ValueError("weights should have the same first two dimensions as X.")

    b, n, dim = Xt.shape

    if (num_points < Xt.shape[1]).any() or (num_points < Yt.shape[1]).any():
        # in case we got Pointclouds as input, mask the unused entries in Xc, Yc
        mask = (
            torch.arange(n, dtype=torch.int64, device=Xt.device)[None]
            < num_points[:, None]
        ).type_as(Xt)
        weights = mask if weights is None else mask * weights.type_as(Xt)

    # compute the centroids of the point sets
    Xmu = wmean(Xt, weight=weights, eps=eps)
    Ymu = wmean(Yt, weight=weights, eps=eps)

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    # ADDED: Scaling
    scales = torch.sqrt(torch.sum(Xc**2, dim=[-2, -1], keepdims=True))
    Xc = Xc / scales

    # scales = torch.sqrt(torch.sum(Yc**2, dim=[-2, -1], keepdims=True))
    Yc = Yc / scales


    total_weight = torch.clamp(num_points, 1)
    # special handling for heterogeneous point clouds and/or input weights
    if weights is not None:
        Xc *= weights[:, :, None]
        Yc *= weights[:, :, None]
        total_weight = torch.clamp(weights.sum(1), eps)

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    # XYcov = XYcov / total_weight[:, None, None]

    # print(XYcov)
    # decompose the covariance matrix XYcov
    U, S, V = torch.svd(XYcov)

    if torch.sum(U) == 0.0:
        raise ValueError

    # print(U, S, V)
    # error

    # catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= AMBIGUOUS_ROT_SINGULAR_THR).any() and not (
        num_points < (dim + 1)
    ).any():
        warnings.warn(
            "Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # identity matrix used for fixing reflections
    E = torch.eye(dim, dtype=XYcov.dtype, device=XYcov.device)[None].repeat(b, 1, 1)

    if not allow_reflection:
        # reflection test:
        #   checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # find the rotation matrix by composing U and V again
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # estimate the scaling component of the transformation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / total_weight

        # the scaling component
        s = trace_ES / torch.clamp(Xcov, eps)

        # translation component
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # translation component
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]

        # unit scaling since we do not estimate scale
        s = T.new_ones(b)

    return SimilarityTransform(R, T, s)


def _apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X


def define_loss(args):
    loss_model = VGG_feat()
    loss_model = torch.nn.DataParallel(loss_model).cuda(args.gpu)
    loss_model = loss_model.eval()

    recon_crit = mse_loss_mask().cuda(args.gpu)
    separation_crit = separation_loss(args.nkpts).cuda(args.gpu)
    rotation_crit = rotation_loss().cuda(args.gpu)

    criterion = [recon_crit, nn.MSELoss().cuda(args.gpu), separation_crit, rotation_crit]

    return loss_model, criterion

class computeLoss:
    def __init__(self, args):
        self.args = args

        self.loss_model = VGG_feat()
        self.loss_model = torch.nn.DataParallel(self.loss_model).cuda(args.gpu)
        self.loss_model = self.loss_model.eval()

        recon_crit = mse_loss_mask().cuda(args.gpu)
        separation_crit = separation_loss(args.nkpts).cuda(args.gpu)
        rotation_crit = rotation_loss().cuda(args.gpu)

        self.distance_criterion = torch.nn.PairwiseDistance()
        self.length_criterion = nn.MSELoss()

        self.criterion = [recon_crit, nn.MSELoss().cuda(args.gpu), separation_crit, rotation_crit]

    def update_loss(self, inputs, tr_inputs, loss_mask, output, epoch):
        device = inputs.device

        to_pred = compute_ssim(inputs, tr_inputs)

        vgg_feat_in = self.loss_model(to_pred)
        vgg_feat_out = self.loss_model(output['recon'])

        l = self.criterion[0](to_pred, output['recon'], loss_mask)
        wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[0])
        l /= wl

        loss = l.mean()

        for _i in range(0, len(vgg_feat_in)):
            _mask = F.upsample(loss_mask,
                               size=(vgg_feat_in[_i].size(2), vgg_feat_in[_i].size(3)), mode='bilinear') # in_mask
            l = self.criterion[0](vgg_feat_in[_i], vgg_feat_out[_i], _mask)
            wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[_i+1])
            l /= wl

            loss += l.mean()


        if 'tr_pos' in output.keys():
            separation = self.criterion[2](output['tr_pos'])
            loss += separation.mean()

            separation = self.criterion[2](output['pos'])
            loss += separation.mean()        

        if epoch >= self.args.curriculum and 'gmtr_heatmap' in output.keys():
            deg = torch.ones((output['gmtr_heatmap'][0].size()[0])).to(device) * 90
            rot_loss, rot_label = self.criterion[3](output['tr_heatmap'][:, :self.args.nkpts], output['gmtr_heatmap'][0], deg)
            loss += rot_loss/3

            deg = torch.ones((output['gmtr_heatmap'][1].size()[0])).to(device) * 180
            rot_loss, rot_label2 = self.criterion[3](output['tr_heatmap'][:, :self.args.nkpts], output['gmtr_heatmap'][1], deg)
            loss += rot_loss/3

            deg = torch.ones((output['gmtr_heatmap'][2].size()[0])).to(device) * -90
            rot_loss, rot_label3 = self.criterion[3](output['tr_heatmap'][:, :self.args.nkpts], output['gmtr_heatmap'][2], deg)
            loss += rot_loss/3

            separation = self.criterion[2]((output['gmtr_pos'][0], output['gmtr_pos'][1]))
            loss += separation.mean()

            separation = self.criterion[2]((output['gmtr_pos'][2], output['gmtr_pos'][3]))
            loss += separation.mean()

            separation = self.criterion[2]((output['gmtr_pos'][4], output['gmtr_pos'][5]))
            loss += separation.mean()

            separation = self.criterion[2](output['tr_pos'])
            loss += separation.mean()

        return loss



    def update_loss_2(self, inputs, tr_inputs, loss_mask, output, epoch):
        device = inputs[0].device

        loss = 0

        kp_ssim_loss = 0

        ssim_list = []

        for view_ind in range(len(inputs)):
            to_pred = compute_ssim(inputs[view_ind], tr_inputs[view_ind])

            ssim_list.append(to_pred)

            vgg_feat_in = self.loss_model(to_pred)
            vgg_feat_out = self.loss_model(output['recon'][view_ind])

            l = self.criterion[0](to_pred, output['recon'][view_ind], loss_mask)
            wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[0])
            l /= wl

            loss += l.mean()

            for _i in range(0, len(vgg_feat_in)):
                _mask = F.upsample(loss_mask,
                                   size=(vgg_feat_in[_i].size(2), vgg_feat_in[_i].size(3)), mode='bilinear') # in_mask
                l = self.criterion[0](vgg_feat_in[_i], vgg_feat_out[_i], _mask)
                wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[_i+1])
                l /= wl

                loss += l.mean()


            # mean_to_pred = torch.mean(to_pred, dim = 1, keepdim = True)[:, :, ::2, ::2]

            # # print(mean_to_pred.size())
            # # print(output['tr_kp_cond'][view_ind].size())

            # # print(mean_to_pred)
            # # print(output['tr_kp_cond'][view_ind])
            # # print(mean_to_pred)            

            # kp_ssim_loss -= (output['tr_kp_cond'][view_ind]*mean_to_pred).mean()*10
            # error


            # ########## 2D recon
            # to_pred = compute_ssim(inputs[view_ind], tr_inputs[view_ind])

            # vgg_feat_in = self.loss_model(to_pred)
            # vgg_feat_out = self.loss_model(output['recon_2d'][view_ind])

            # l = self.criterion[0](to_pred, output['recon_2d'][view_ind], loss_mask)
            # wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[0])
            # l /= wl

            # loss += l.mean()

            # for _i in range(0, len(vgg_feat_in)):
            #     _mask = F.upsample(loss_mask,
            #                        size=(vgg_feat_in[_i].size(2), vgg_feat_in[_i].size(3)), mode='bilinear') # in_mask
            #     l = self.criterion[0](vgg_feat_in[_i], vgg_feat_out[_i], _mask)
            #     wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[_i+1])
            #     l /= wl

            #     loss += l.mean()


            ### Penalize 2D poses not on SSIM image


        # if 'tr_pos' in output.keys():
        separation = self.criterion[2].loss_3d(output['tr_pos_3d'])
        loss += separation.mean()

        separation = self.criterion[2].loss_3d(output['pos_3d'])
        loss += separation.mean()        

        # if epoch >= self.args.curriculum and 'gmtr_heatmap' in output.keys():
        #     deg = torch.ones((output['gmtr_heatmap'][0].size()[0])).to(device) * 90
        #     rot_loss, rot_label = self.criterion[3](output['tr_heatmap'][:, :self.args.nkpts], output['gmtr_heatmap'][0], deg)
        #     loss += rot_loss/3

        #     deg = torch.ones((output['gmtr_heatmap'][1].size()[0])).to(device) * 180
        #     rot_loss, rot_label2 = self.criterion[3](output['tr_heatmap'][:, :self.args.nkpts], output['gmtr_heatmap'][1], deg)
        #     loss += rot_loss/3

        #     deg = torch.ones((output['gmtr_heatmap'][2].size()[0])).to(device) * -90
        #     rot_loss, rot_label3 = self.criterion[3](output['tr_heatmap'][:, :self.args.nkpts], output['gmtr_heatmap'][2], deg)
        #     loss += rot_loss/3

        #     separation = self.criterion[2]((output['gmtr_pos'][0], output['gmtr_pos'][1]))
        #     loss += separation.mean()

        #     separation = self.criterion[2]((output['gmtr_pos'][2], output['gmtr_pos'][3]))
        #     loss += separation.mean()

        #     separation = self.criterion[2]((output['gmtr_pos'][4], output['gmtr_pos'][5]))
        #     loss += separation.mean()

        #     separation = self.criterion[2](output['tr_pos'])
        #     loss += separation.mean()

        # loss += kp_ssim_loss

        # print(kp_ssim_loss)


        return loss, ssim_list


    def update_loss_3(self, inputs, tr_inputs, edge_weights, loss_mask, output, epoch, num_kpts):
        device = inputs[0].device

        loss = 0

        kp_ssim_loss = 0

        ssim_list = []

        for view_ind in range(len(inputs)):
            to_pred = compute_ssim(inputs[view_ind], tr_inputs[view_ind])

            ssim_list.append(to_pred)

            vgg_feat_in = self.loss_model(to_pred)
            vgg_feat_out = self.loss_model(output['recon'][view_ind])

            l = self.criterion[0](to_pred, output['recon'][view_ind], loss_mask)
            wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[0])
            l /= wl

            loss += l.mean()

            for _i in range(0, len(vgg_feat_in)):
                _mask = F.upsample(loss_mask,
                                   size=(vgg_feat_in[_i].size(2), vgg_feat_in[_i].size(3)), mode='bilinear') # in_mask
                l = self.criterion[0](vgg_feat_in[_i], vgg_feat_out[_i], _mask)
                wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[_i+1])
                l /= wl

                loss += l.mean()


        # if 'tr_pos' in output.keys():
        separation = self.criterion[2].loss_3d(output['tr_pos_3d'])
        loss += separation.mean()

        separation = self.criterion[2].loss_3d(output['pos_3d'])
        loss += separation.mean()        


        boolean_edge = edge_weights.clone().detach() > 0.0

        # size 105

        pairs = np.array(list(itertools.combinations(range(num_kpts), 2)))


        # 2 x 15 x 3
        batch_size = output['pos_3d'].size()[0]
        distances = self.distance_criterion(output['pos_3d'][:, pairs[:, 0], :].view(-1, 3),
            output['pos_3d'][:, pairs[:, 1], :].view(-1, 3))

        distances = distances.view(batch_size, -1)

        distances = torch.einsum('mb,b->mb', distances, boolean_edge)


        distances_2 = self.distance_criterion(output['tr_pos_3d'][:, pairs[:, 0], :].view(-1, 3),
            output['tr_pos_3d'][:, pairs[:, 1], :].view(-1, 3))

        distances_2 = distances_2.view(batch_size, -1)

        distances_2 = torch.einsum('mb,b->mb', distances_2, boolean_edge)


        all_dist = torch.cat([distances, distances_2], dim = 0)
        mean_distance = torch.mean(all_dist, dim = 0, keepdim = True).repeat(all_dist.size()[0],  1)

        length_loss = self.length_criterion(mean_distance[:, boolean_edge], all_dist[:, boolean_edge])

        # loss += length_loss/10000
        # print(distances, distances_2)
        # print(length_loss)

        # Penalize disconnected edges?

        

        return loss, length_loss, ssim_list



    def update_loss_4(self, inputs, tr_inputs, running_average, edge_weights, loss_mask, output, epoch, num_kpts):
        device = inputs[0].device

        loss = 0

        kp_ssim_loss = 0

        ssim_list = []

        for view_ind in range(len(inputs)):
            to_pred = compute_ssim(inputs[view_ind], tr_inputs[view_ind])

            ssim_list.append(to_pred)

            vgg_feat_in = self.loss_model(to_pred)
            vgg_feat_out = self.loss_model(output['recon'][view_ind])

            l = self.criterion[0](to_pred, output['recon'][view_ind], loss_mask)
            wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[0])
            l /= wl

            loss += l.mean()

            for _i in range(0, len(vgg_feat_in)):
                _mask = F.upsample(loss_mask,
                                   size=(vgg_feat_in[_i].size(2), vgg_feat_in[_i].size(3)), mode='bilinear') # in_mask
                l = self.criterion[0](vgg_feat_in[_i], vgg_feat_out[_i], _mask)
                wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[_i+1])
                l /= wl

                loss += l.mean()

        print(loss)


        # for view_ind in range(len(inputs)):
        #     to_pred = compute_ssim(inputs[view_ind], tr_inputs[view_ind])

        #     vgg_feat_in = self.loss_model(to_pred)
        #     vgg_feat_out = self.loss_model(output['recon2'][view_ind])

        #     l = self.criterion[0](to_pred, output['recon2'][view_ind], loss_mask)
        #     wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[0])
        #     l /= wl

        #     loss += l.mean()

        #     for _i in range(0, len(vgg_feat_in)):
        #         _mask = F.upsample(loss_mask,
        #                            size=(vgg_feat_in[_i].size(2), vgg_feat_in[_i].size(3)), mode='bilinear') # in_mask
        #         l = self.criterion[0](vgg_feat_in[_i], vgg_feat_out[_i], _mask)
        #         wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[_i+1])
        #         l /= wl

        #         loss += l.mean()

        # print(loss)


        # if 'tr_pos' in output.keys():
        separation = self.criterion[2].loss_3d(output['tr_pos_3d'])
        loss += separation.sum()

        separation = self.criterion[2].loss_3d(output['pos_3d'])
        loss += separation.sum()        

        print(separation.sum())

        boolean_edge = edge_weights.detach() > 0.0

        # size 105

        pairs = np.array(list(itertools.combinations(range(num_kpts), 2)))


        # 2 x 15 x 3
        batch_size = output['pos_3d'].size()[0]
        distances = self.distance_criterion(output['pos_3d'][:, pairs[:, 0], :].view(-1, 3),
            output['pos_3d'][:, pairs[:, 1], :].view(-1, 3))

        distances = distances.view(batch_size, -1)

        distances = torch.einsum('mb,b->mb', distances, boolean_edge)


        distances_2 = self.distance_criterion(output['tr_pos_3d'][:, pairs[:, 0], :].view(-1, 3),
            output['tr_pos_3d'][:, pairs[:, 1], :].view(-1, 3))

        distances_2 = distances_2.view(batch_size, -1)

        distances_2 = torch.einsum('mb,b->mb', distances_2, boolean_edge)


        all_dist = torch.cat([distances, distances_2], dim = 0)
        mean_distance = torch.mean(all_dist, dim = 0) #.repeat(all_dist.size()[0],  1)

        # running_average = running_average.repeat(all_dist.size()[0],  1)

        # print(mean_distance.size())
        # print(running_average.size())
        # error

        if torch.sum(running_average) == 0:
            running_average = mean_distance
        else:
            running_average = _exp_running_avg(mean_distance, init_val = running_average)

        length_loss = 0
        length_loss += self.length_criterion(running_average[boolean_edge].unsqueeze(0).detach(), distances[:, boolean_edge])
        length_loss += self.length_criterion(running_average[boolean_edge].unsqueeze(0).detach(), distances_2[:, boolean_edge])


        print('Length', length_loss)
        # print(running_average)

        # #########################################################        
        # try:
        #     # print(output['pos_3d'].size())
        #     aligned = corresponding_points_alignment(output['pos_3d'], output['tr_pos_3d'])
        #     # error

        #     # print(_apply_similarity_transform(output['pos_3d'], aligned.R, aligned.T, aligned.s))
        #     # error
        #     difference = torch.norm(_apply_similarity_transform(output['pos_3d'], aligned.R, aligned.T, aligned.s) - \
        #         output['tr_pos_3d'], dim = -1).mean()

        #     print('Aligned', difference)

        #     loss += difference/1000

        #     del aligned
        # except:
        #     pass

        running_average = running_average
        return loss, length_loss, ssim_list, running_average



# x_avg: torch variable which is initialized to init_val - weight
def _exp_running_avg(x, rho=0.99, init_val=0.0):
    x_avg = init_val

    w_update = 1.0 - rho
    x_new = x_avg + w_update * (x - x_avg)
    return x_new

## Sum all losses here
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.separation_loss import separation_loss
from loss.rotation_loss import rotation_loss
from loss.ssim_loss import compute_ssim

from model.vggforLoss import mse_loss_mask, VGG_feat

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

        for view_ind in range(len(inputs)):
            to_pred = compute_ssim(inputs[view_ind], tr_inputs[view_ind])

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


        return loss


# x_avg: torch variable which is initialized to init_val - weight
def _exp_running_avg(x, rho=0.99, init_val=0.0):
    x_avg = init_val

    w_update = 1.0 - rho
    x_new = x_avg + w_update * (x - x_avg)
    return x_new

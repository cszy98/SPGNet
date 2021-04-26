from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
import argparse

from . import SPG_net_market
from . import SPG_net_deepfashion
from . import networks
from .base_model import BaseModel
from util import io, pose_util

class PoseTransferModel(BaseModel):
    '''
    Pose transfer framework that cascade a 3d-flow module and a generation module.
    '''

    def name(self):
        return 'PoseTransferModel'

    def initialize(self, opt):
        super(PoseTransferModel, self).initialize(opt)
        ###################################
        # define generator
        ###################################
        self.seg_cihp_nc = 20
        self.use_parsing = True
        self.opt = opt
        if opt.dataset_name == 'market':
            SPGNet = SPG_net_market
        else:
            SPGNet = SPG_net_deepfashion
        self.netG = SPGNet.DualUnetGenerator_SEAN(
                    pose_nc=self.get_tensor_dim(opt.G_pose_type),
                    appearance_nc=self.get_tensor_dim(opt.G_appearance_type),
                    output_nc=3,
                    aux_output_nc=[],
                    nf=opt.G_nf,
                    max_nf=opt.G_max_nf,
                    num_scales=opt.G_n_scale,
                    num_warp_scales=opt.G_n_warp_scale,
                    n_residual_blocks=2,
                    norm=opt.G_norm,
                    vis_mode=opt.G_vis_mode,
                    activation=nn.LeakyReLU(0.1) if opt.G_activation == 'leaky_relu' else nn.ReLU(),
                    use_dropout=opt.use_dropout,
                    no_end_norm=opt.G_no_end_norm,
                    gpu_ids=opt.gpu_ids,
                    isTrain = self.is_train
        )
        # print(self.netG)
        if opt.gpu_ids:
            self.netG.cuda()
        networks.init_weights(self.netG, init_type=opt.init_type)
        ###################################
        # define external pixel warper
        ###################################
        if opt.G_pix_warp:
            pix_warp_n_scale = opt.G_n_scale
            self.netPW = networks.UnetGenerator_MultiOutput(
                input_nc=self.get_tensor_dim(opt.G_pix_warp_input_type),
                output_nc=[1],  # only use one output branch (weight mask)
                nf=32,
                max_nf=128,
                num_scales=pix_warp_n_scale,
                n_residual_blocks=2,
                norm=opt.G_norm,
                activation=nn.ReLU(False),
                use_dropout=False,
                gpu_ids=opt.gpu_ids
            )
            if opt.gpu_ids:
                self.netPW.cuda()
            networks.init_weights(self.netPW, init_type=opt.init_type)
        ###################################
        # define discriminator
        ###################################
        self.use_gan = self.is_train and self.opt.loss_weight_gan > 0
        if self.use_gan:
            self.netD = networks.NLayerDiscriminator(
                input_nc=self.get_tensor_dim(opt.D_input_type_real),
                ndf=opt.D_nf,
                n_layers=opt.D_n_layers,
                use_sigmoid=(opt.gan_type == 'dcgan'),
                output_bias=True,
                gpu_ids=opt.gpu_ids,
            )
            if opt.gpu_ids:
                self.netD.cuda()
            networks.init_weights(self.netD, init_type=opt.init_type)
        ###################################
        # load optical flow model
        ###################################
        if opt.flow_on_the_fly:
            self.netF = load_flow_network(opt.pretrained_flow_id, opt.pretrained_flow_epoch, opt.gpu_ids)
            self.netF.eval()
            if opt.gpu_ids:
                self.netF.cuda()

        ###################################
        # loss and optimizers
        ###################################
        # self.crit_psnr = networks.PSNR().cuda()
        self.crit_ssim = networks.SSIM().cuda()

        if self.is_train:
            self.crit_vgg = networks.VGGLoss(opt.gpu_ids, shifted_style=opt.shifted_style_loss,
                                             content_weights=opt.vgg_content_weights)
            if opt.G_pix_warp:
                # only optimze netPW
                self.optim = torch.optim.Adam(self.netPW.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay)
            else:
                self.optim = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
                                              weight_decay=opt.weight_decay)
            self.optimizers = [self.optim]

            if self.use_gan:
                self.crit_gan = networks.GANLoss(use_lsgan=(opt.gan_type == 'lsgan'))
                if self.gpu_ids:
                    self.crit_gan.cuda()
                self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2),
                                                weight_decay=opt.weight_decay_D)
                self.optimizers += [self.optim_D]

        ###################################
        # load trained model
        ###################################
        if not self.is_train:
            # load trained model for testing
            self.load_network(self.netG, 'netG', opt.which_epoch)
            if opt.G_pix_warp:
                self.load_network(self.netPW, 'netPW', opt.which_epoch)
        elif opt.pretrained_G_id is not None:
            # load pretrained network
            self.load_network(self.netG, 'netG', opt.pretrained_G_epoch, opt.pretrained_G_id)
        elif opt.resume_train:
            # resume training
            self.load_network(self.netG, 'netG', opt.last_epoch)
            self.load_optim(self.optim, 'optim', opt.last_epoch)
            # note
            if self.use_gan:
                self.load_network(self.netD, 'netD', opt.last_epoch)
                self.load_optim(self.optim_D, 'optim_D', opt.last_epoch)
            if opt.G_pix_warp:
                self.load_network(self.netPW, 'netPW', opt.last_epoch)
        ###################################
        # schedulers
        ###################################
        if self.is_train:
            self.schedulers = []
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))

    def set_input(self, data):
        self.input_list = [
            'img_1',
            'img_2',
            'joint_1',
            'joint_2',
        ]
        if self.use_parsing:
            self.input_list += ['seg_cihp_1',
                                'seg_cihp_2'
                                ]

        for item in self.input_list:
            self.input[item] = self.Tensor(data[item].size()).copy_(data[item])

        self.input['id'] = zip(data['id_1'], data['id_2'])

    def forward(self, test=False):
        # generate flow
        flow_scale = 20.
        if self.opt.flow_on_the_fly:
            with torch.no_grad():
                input_F = self.get_tensor(self.opt.F_input_type)
                flow_out, vis_out, _, _ = self.netF(input_F)
                self.output['vis_out'] = vis_out.argmax(dim=1, keepdim=True).float()
                self.output['mask_out'] = (self.output['vis_out'] < 2).float()
                self.output['flow_out'] = flow_out * flow_scale * self.output['mask_out']
        else:
            self.output['flow_out'] = self.input['flow_2to1']
            self.output['vis_out'] = self.input['vis_2']
            self.output['mask_out'] = (self.output['vis_out'] < 2).float()
            self.output['flow_tar'] = self.output['flow_out']
            self.output['vis_tar'] = self.output['vis_out']
            self.output['maks_tar'] = self.output['mask_out']
        bsz, _, h, w = self.output['vis_out'].size()
        self.output['vismap_out'] = self.output['vis_out'].new(bsz, 3, h, w).scatter_(dim=1, index=self.output[
            'vis_out'].long(), value=1)

        # warp image
        self.output['img_warp'] = networks.warp_acc_flow(self.input['img_1'], self.output['flow_out'],
                                                         mask=self.output['mask_out'])

        # generate image
        if self.opt.which_model_G == 'unet':
            input_G = self.get_tensor('+'.join([self.opt.G_appearance_type, self.opt.G_pose_type]))
            out = self.netG(input_G)
            self.output['img_out'] = F.tanh(out)
        elif self.opt.which_model_G == 'dual_unet':
            input_G_pose = self.get_tensor(self.opt.G_pose_type)
            input_G_appearance = self.get_tensor(self.opt.G_appearance_type)
            input_G_s_seg = self.get_tensor('seg_cihp_1')

            input_G_d_seg = self.get_tensor('seg_cihp_2')
            flow_in, vis_in = (self.output['flow_out'], self.output['vis_out']) if self.opt.G_feat_warp else (
            None, None)

            dismap = None
            if not self.opt.G_pix_warp:
                out = self.netG(input_G_pose, input_G_appearance, input_G_s_seg, input_G_d_seg, flow_in, vis_in, dismap)
                self.output['img_out'] = F.tanh(out)
            else:
                with torch.no_grad():
                    out = self.netG(input_G_pose, input_G_appearance, input_G_s_seg, input_G_d_seg, flow_in, vis_in)
                self.output['img_out_G'] = F.tanh(out)
                pw_out = self.netPW(self.get_tensor(self.opt.G_pix_warp_input_type))
                self.output['pix_mask'] = F.sigmoid(pw_out[0])
                if self.opt.G_pix_warp_detach:
                    self.output['img_out'] = self.output['img_warp'] * self.output['pix_mask'] + self.output[
                        'img_out_G'].detach() * (1 - self.output['pix_mask'])
                else:
                    self.output['img_out'] = self.output['img_warp'] * self.output['pix_mask'] + self.output[
                        'img_out_G'] * (1 - self.output['pix_mask'])
        self.output['img_tar'] = self.input['img_2']

    def test(self, compute_loss=True, meas_only=True):
        ''' meas_only: only compute measurements (psrn, ssim) when computing loss'''
        with torch.no_grad():
            self.forward(test=True)
            if compute_loss:
                assert self.is_train or meas_only, 'when is_train is False, meas_only must be True'
                self.compute_loss(meas_only=meas_only, compute_ssim=True)

    def compute_loss(self, meas_only=False, compute_ssim=False):
        '''compute_ssim: set True to compute ssim (time consuming)'''
        ##############################
        # measurements
        ##############################
        if compute_ssim:
            self.output['SSIM'] = self.crit_ssim(self.output['img_out'], self.output['img_tar'])
        if meas_only:
            return
        ##############################
        # losses
        ##############################
        self.output['loss_l1'] = F.l1_loss(self.output['img_out'], self.output['img_tar'])
        # Content (Perceptual)
        self.output['loss_content'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], loss_type='content')
        # Style
        if self.opt.loss_weight_style > 0:
            self.output['loss_style'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], loss_type='style')
        # GAN
        if self.use_gan:
            input_D = self.get_tensor(self.opt.D_input_type_fake)
            self.output['loss_G'] = self.crit_gan(self.netD(input_D), True)

    def backward(self, check_grad=False):
        loss_ce = 0.5
        # if not check_grad:
        loss = 0
        loss += self.output['loss_l1'] * self.opt.loss_weight_l1
        loss += self.output['loss_content'] * self.opt.loss_weight_content
        if self.opt.loss_weight_style > 0:
            loss += self.output['loss_style'] * self.opt.loss_weight_style
        if self.use_gan:
            loss += self.output['loss_G'] * self.opt.loss_weight_gan

        self.output['total_G_loss'] = loss
        loss.backward()

    def backward_D(self):
        input_D_real = self.get_tensor(self.opt.D_input_type_real).detach()
        input_D_fake = self.get_tensor(self.opt.D_input_type_fake).detach()
        self.output['loss_D'] = 0.5 * (self.crit_gan(self.netD(input_D_real), True) + \
                                       self.crit_gan(self.netD(input_D_fake), False))
        (self.output['loss_D'] * self.opt.loss_weight_gan).backward()

    def optimize_parameters(self, check_grad=False):
        self.output = {}
        # forward
        self.forward()
        # optim netD
        if self.use_gan:
            self.optim_D.zero_grad()
            self.backward_D()
            self.optim_D.step()
        # optim netG
        self.optim.zero_grad()
        self.compute_loss()
        self.backward(check_grad)
        self.optim.step()

    def get_tensor_dim(self, tensor_type):
        dim = 0
        tensor_items = tensor_type.split('+')
        for item in tensor_items:
            if item in {'img_1', 'img_2', 'img_out', 'img_warp', 'img_out_G'}:
                dim += 3
            elif item in {'seg_1', 'seg_2'}:
                dim += self.opt.seg_nc
            elif item in {'seg_cihp_1', 'seg_cihp_2', 'pre_seg_cihp_2'}:
                dim += self.seg_cihp_nc
            elif item in {'joint_1', 'joint_2'}:
                dim += self.opt.joint_nc
            elif item in {'flow_out', 'flow_tar'}:
                dim += 2
            elif item in {'vis_out', 'vis_tar'}:
                dim += 1
            elif item in {'vismap_out', 'vismap_tar'}:
                dim += 3
            else:
                raise Exception('invalid tensor_type: %s' % item)
        return dim

    def get_tensor(self, tensor_type):
        tensor = []
        tensor_items = tensor_type.split('+')
        for item in tensor_items:
            if item == 'img_1':
                tensor.append(self.input['img_1'])
            elif item == 'img_2':
                tensor.append(self.input['img_2'])
            elif item == 'img_out':
                tensor.append(self.output['img_out'])
            elif item == 'img_out_G':
                tensor.append(self.output['img_out_G'])
            elif item == 'img_warp':
                tensor.append(self.output['img_warp'])
            elif item == 'seg_1':
                tensor.append(self.input['seg_1'])
            elif item == 'seg_2':
                tensor.append(self.input['seg_2'])
            elif item == 'seg_cihp_1':
                tensor.append(self.input['seg_cihp_1'])
            elif item == 'seg_cihp_2':
                tensor.append(self.input['seg_cihp_2'])
            elif item == 'pre_seg_cihp_2':
                tensor.append(self.output['pre_seg_cihp_2'])
            elif item == 'joint_1':
                tensor.append(self.input['joint_1'])
            elif item == 'joint_2':
                tensor.append(self.input['joint_2'])
            elif item == 'flow_out':
                tensor.append(self.output['flow_out'])
            elif item == 'flow_tar':
                tensor.append(self.output['flow_tar'])
            elif item == 'vis_out':
                tensor.append(self.output['vis_out'])
            elif item == 'vis_tar':
                tensor.append(self.output['vis_tar'])
            elif item == 'vismap_out':
                tensor.append(self.output['vismap_out'])
            elif item == 'vismap_tar':
                tensor.append(self.output['vismap_tar'])
            elif item == 'dis_map':
                if self.opt.joint_PATN:
                    self.input['dis_map'] = torch.exp((-0.1) * self.input['dis_map'])
                tensor.append(self.input['dis_map'])
            else:
                raise Exception('invalid tensor_type: %s' % item)
        tensor = torch.cat(tensor, dim=1)
        return tensor

    def get_current_errors(self):
        error_list = [
            'PSNR',
            'SSIM',
            'mask_SSIM',
            'loss_l1',
            'loss_content',
            'loss_style',
            'loss_G',
            'loss_D',
            'total_G_loss',
            'grad_l1',
            'grad_content',
            'grad_style',
            'grad_G',
            'loss_ce'
        ]
        errors = OrderedDict()
        for item in error_list:
            if item in self.output:
                errors[item] = self.output[item].item()

        return errors

    def delvar(self):
        for k in self.output.keys():
            del self.output[k]
        torch.cuda.empty_cache()

    def get_current_visuals(self):
        visual_items = [
            ('img_1', [self.input['img_1'].data.cpu(), 'rgb']),
            ('joint_1', [self.input['joint_1'].data.cpu(), 'pose']),
            ('joint_2', [self.input['joint_2'].data.cpu(), 'pose']),
            ('flow_out', [self.output['flow_out'].data.cpu(), 'flow']),
            ('vis_out', [self.output['vis_out'].data.cpu(), 'vis']),
        ]

        if self.use_parsing:
            visual_items += [
                ('seg_cihp_1', [self.input['seg_cihp_1'].data.cpu(), 'seg']),
                ('seg_cihp_2', [self.input['seg_cihp_2'].data.cpu(), 'seg'])
            ]

        if self.opt.G_pix_warp:
            visual_items += [
                ('img_warp', [self.output['img_warp'].data.cpu(), 'rgb']),
                ('img_out_G', [self.output['img_out_G'].data.cpu(), 'rgb']),
                ('pix_mask', [self.output['pix_mask'].data.cpu(), 'softmask']),
                ('img_out', [self.output['img_out'].data.cpu(), 'rgb']),
                ('img_tar', [self.output['img_tar'].data.cpu(), 'rgb'])
            ]
        else:
            visual_items += [
                ('img_warp', [self.output['img_warp'].data.cpu(), 'rgb']),
                ('img_out', [self.output['img_out'].data.cpu(), 'rgb']),
                ('img_tar', [self.output['img_tar'].data.cpu(), 'rgb'])
            ]

        visuals = OrderedDict(visual_items)
        return visuals

    def save(self, label):
        # save network weights
        self.save_network(self.netG, 'netG', label, self.gpu_ids)
        if self.opt.G_pix_warp:
            self.save_network(self.netPW, 'netPW', label, self.gpu_ids)
        # save optimizer status
        self.save_optim(self.optim, 'optim', label)
        # note
        self.save_network(self.netD, 'netD', label, self.gpu_ids)
        self.save_optim(self.optim_D, 'optim_D', label)

    def train(self):
        # netG and netD will always be in 'train' status
        pass

    def eval(self):
        # netG and netD will always be in 'train' status
        pass


##################################################
# helper functions
##################################################
def load_flow_network(model_id, epoch='best', gpu_ids=[]):
    from .flow_regression_model import FlowRegressionModel
    opt_dict = io.load_json(os.path.join('checkpoints', model_id, 'train_opt.json'))
    opt = argparse.Namespace(**opt_dict)
    opt.gpu_ids = gpu_ids
    opt.is_train = False  # prevent loading discriminator, optimizer...
    opt.which_epoch = epoch
    # create network
    model = FlowRegressionModel()
    model.initialize(opt)
    return model.netF
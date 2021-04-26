import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
import os
import numpy as np
import re
import torch.nn.utils.spectral_norm as spectral_norm


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
        norm_layer(out_channels),
    )
    return model

def channel_mapping(in_channels, out_channels, norm_layer=nn.BatchNorm2d, bias=False):
    return conv(in_channels, out_channels, kernel_size=1, norm_layer=norm_layer, bias=bias)

###############################################################################
# flow-based warping
###############################################################################
def warp_acc_flow(x, flow, mode='bilinear', mask=None, mask_value=-1):
    '''
    warp an image/tensor according to given flow.
    Input:
        x: (bsz, c, h, w)
        flow: (bsz, c, h, w)
        mask: (bsz, 1, h, w). 1 for valid region and 0 for invalid region. invalid region will be fill with "mask_value" in the output images.
    Output:
        y: (bsz, c, h, w)
    '''
    bsz, c, h, w = x.size()
    # mesh grid
    xx = x.new_tensor(range(w)).view(1,-1).repeat(h,1)
    yy = x.new_tensor(range(h)).view(-1,1).repeat(1,w)
    xx = xx.view(1,1,h,w).repeat(bsz,1,1,1)
    yy = yy.view(1,1,h,w).repeat(bsz,1,1,1)
    grid = torch.cat((xx,yy), dim=1).float()
    grid = grid + flow
    # scale to [-1, 1]
    grid[:,0,:,:] = 2.0*grid[:,0,:,:]/max(w-1,1) - 1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:]/max(h-1,1) - 1.0

    grid = grid.permute(0,2,3,1)
    output = F.grid_sample(x, grid, mode=mode, padding_mode='zeros')
    # mask = F.grid_sample(x.new_ones(x.size()), grid)
    # mask = torch.where(mask<0.9999, mask.new_zeros(1), mask.new_ones(1))
    # return output * mask
    if mask is not None:
        output = torch.where(mask>0.5, output, output.new_ones(1).mul_(mask_value))
    return output

class Identity(nn.Module):
    def __init__(self, dim=None):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class ResidualBlock(nn.Module):
    '''
    Derived from Variational UNet.
    '''

    def __init__(self, dim, dim_a, norm_layer=nn.BatchNorm2d, use_bias=False, activation=nn.ReLU(False),
                 use_dropout=False, no_end_norm=False):
        super(ResidualBlock, self).__init__()
        self.use_dropout = use_dropout
        self.activation = activation
        if dim_a is None or dim_a <= 0:
            # w/o additional input
            if no_end_norm:
                self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity,
                                 bias=True)
            else:
                self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=norm_layer,
                                 bias=use_bias)
        else:
            # w/ additional input
            self.conv_a = channel_mapping(in_channels=dim_a, out_channels=dim, norm_layer=norm_layer, bias=use_bias)
            if no_end_norm:
                self.conv = conv(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity,
                                 bias=True)
            else:
                self.conv = conv(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, norm_layer=norm_layer,
                                 bias=use_bias)

    def forward(self, x, a=None):
        if a is None:
            # w/o additional input
            residual = x
        else:
            # w/ additional input
            a = self.conv_a(self.activation(a))
            residual = torch.cat((x, a), dim=1)
        residual = self.conv(self.activation(residual))
        out = x + residual
        if self.use_dropout:
            out = F.dropout(out, p=0.5, training=self.training)
        return out

class GateBlock(nn.Module):
    def __init__(self, dim, dim_a, activation=nn.ReLU(False)):
        super(GateBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=dim_a, out_channels=dim, kernel_size=1)

    def forward(self, x, a):
        '''
        x: (bsz, dim, h, w)
        a: (bsz, dim_a, h, w)
        '''
        a = self.activation(a)
        g = F.sigmoid(self.conv(a))
        return x * g

#################################################

class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 64

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, segmap):

        inputmap = segmap

        actv = self.mlp_shared(inputmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return gamma, beta

class Zencoder(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=2, norm_layer=nn.InstanceNorm2d):
        super(Zencoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                 norm_layer(ngf), nn.LeakyReLU(0.2, False)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, False)]

        ### upsample
        for i in range(1):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.LeakyReLU(0.2, False)]

        model += [nn.ReflectionPad2d(1), nn.Conv2d(256, output_nc, kernel_size=3, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


    def forward(self, input, segmap):

        codes = self.model(input)

        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        b_size = codes.shape[0]
        # h_size = codes.shape[2]
        # w_size = codes.shape[3]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)


        for i in range(b_size):
            for j in range(s_size):
                # segmap_bool=segmap.type(torch.uint8)
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

                    # codes_avg[i].masked_scatter_(segmap.bool()[i, j], codes_component_mu)

        return codes_vector

class ACE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, ACE_Name=None, status='train', spade_params=None, use_rgb=True):
        super().__init__()

        self.ACE_Name = ACE_Name
        self.status = status
        self.save_npy = True
        self.Spade = SPADE(*spade_params)
        self.use_rgb = use_rgb
        self.style_length = 128
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)


        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.


        if self.use_rgb:
            self.create_gamma_beta_fc_layers()

            self.conv_gamma = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)
            self.conv_beta = nn.Conv2d(self.style_length, norm_nc, kernel_size=ks, padding=pw)


    def forward(self, x, segmap, style_codes=None, obj_dic=None):

        # Part 1. generate parameter-free normalized activations
        added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).cuda() * self.noise_var).transpose(1, 3)
        normalized = self.param_free_norm(x+added_noise)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if self.use_rgb:
            [b_size, f_size, h_size, w_size] = normalized.shape
            middle_avg = torch.zeros((b_size, self.style_length, h_size, w_size), device=normalized.device)

            if self.status == 'UI_mode':
                ############## hard coding

                for i in range(1):
                    for j in range(segmap.shape[1]):

                        component_mask_area = torch.sum(segmap.bool()[i, j])

                        if component_mask_area > 0:
                            if obj_dic is None:
                                print('wrong even it is the first input')
                            else:
                                style_code_tmp = obj_dic[str(j)]['ACE']

                                middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_code_tmp))
                                component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length,component_mask_area)

                                middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)

            else:

                for i in range(b_size):
                    for j in range(segmap.shape[1]):
                        # segmap_bool = segmap.type(torch.uint8)
                        component_mask_area = torch.sum(segmap.bool()[i, j])

                        if component_mask_area > 0:


                            middle_mu = F.relu(self.__getattr__('fc_mu' + str(j))(style_codes[i][j]))
                            component_mu = middle_mu.reshape(self.style_length, 1).expand(self.style_length, component_mask_area)

                            middle_avg[i].masked_scatter_(segmap.bool()[i, j], component_mu)


                            if self.status == 'test' and self.save_npy and self.ACE_Name=='up_2_ACE_0':
                                tmp = style_codes[i][j].cpu().numpy()
                                dir_path = 'styles_test'

                                ############### some problem with obj_dic[i]

                                im_name = os.path.basename(obj_dic[i])
                                folder_path = os.path.join(dir_path, 'style_codes', im_name, str(j))
                                if not os.path.exists(folder_path):
                                    os.makedirs(folder_path)

                                style_code_path = os.path.join(folder_path, 'ACE.npy')
                                np.save(style_code_path, tmp)


            gamma_avg = self.conv_gamma(middle_avg)
            beta_avg = self.conv_beta(middle_avg)


            gamma_spade, beta_spade = self.Spade(segmap)

            gamma_alpha = F.sigmoid(self.blending_gamma)
            beta_alpha = F.sigmoid(self.blending_beta)

            gamma_final = gamma_alpha * gamma_avg + (1 - gamma_alpha) * gamma_spade
            beta_final = beta_alpha * beta_avg + (1 - beta_alpha) * beta_spade
            out = normalized * (1 + gamma_final) + beta_final
        else:
            gamma_spade, beta_spade = self.Spade(segmap)
            gamma_final = gamma_spade
            beta_final = beta_spade
            out = normalized * (1 + gamma_final) + beta_final

        return out

    def create_gamma_beta_fc_layers(self):


        ###################  These codes should be replaced with torch.nn.ModuleList

        style_length = self.style_length
        # print(style_length)
        self.fc_mu0 = nn.Linear(style_length, style_length)
        self.fc_mu1 = nn.Linear(style_length, style_length)
        self.fc_mu2 = nn.Linear(style_length, style_length)
        self.fc_mu3 = nn.Linear(style_length, style_length)
        self.fc_mu4 = nn.Linear(style_length, style_length)
        self.fc_mu5 = nn.Linear(style_length, style_length)
        self.fc_mu6 = nn.Linear(style_length, style_length)
        self.fc_mu7 = nn.Linear(style_length, style_length)
        self.fc_mu8 = nn.Linear(style_length, style_length)
        self.fc_mu9 = nn.Linear(style_length, style_length)
        self.fc_mu10 = nn.Linear(style_length, style_length)
        self.fc_mu11 = nn.Linear(style_length, style_length)
        self.fc_mu12 = nn.Linear(style_length, style_length)
        self.fc_mu13 = nn.Linear(style_length, style_length)
        self.fc_mu14 = nn.Linear(style_length, style_length)
        self.fc_mu15 = nn.Linear(style_length, style_length)
        self.fc_mu16 = nn.Linear(style_length, style_length)
        self.fc_mu17 = nn.Linear(style_length, style_length)
        self.fc_mu18 = nn.Linear(style_length, style_length)
        self.fc_mu19 = nn.Linear(style_length, style_length)


class ResidualBlock_SEAN(nn.Module):
    '''
    Derived from Variational UNet.
    '''

    def __init__(self, dim, dim_a, norm_layer=nn.BatchNorm2d, use_bias=False, activation=nn.ReLU(False),
                 use_dropout=False, no_end_norm=False, nc_cihp=20,status='train'):
        super(ResidualBlock_SEAN, self).__init__()
        self.use_dropout = use_dropout
        self.activation = activation
        self.nc_cihp = nc_cihp
        our_norm_type = 'spadebatch5x5'
        Block_Name='SEAN'
        use_rgb=True
        self.status = status

        norm_layer = Identity

        if dim_a is None or dim_a <= 0:
            # w/o additional input
            if no_end_norm:
                self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity,
                                 bias=True)
                self.conv_norm=ACE(our_norm_type, dim, 3, ACE_Name= Block_Name + '_ACE_0', status=self.status,
                                   spade_params=[our_norm_type, dim, nc_cihp], use_rgb=use_rgb)
            else:
                self.conv = conv(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, norm_layer=norm_layer,
                                 bias=use_bias)
                self.conv_norm = ACE(our_norm_type, dim, 3, ACE_Name=Block_Name + '_ACE_0', status=self.status,
                                     spade_params=[our_norm_type, dim, nc_cihp], use_rgb=use_rgb)
        else:
            # w/ additional input
            self.conv_a = spectral_norm(nn.Conv2d(in_channels=dim_a, out_channels=dim, kernel_size=1, bias=use_bias))
            # self.conv_a_norm = SPADE('spadebatch5x5', dim_a, self.nc_cihp)
            self.conv_a_norm = ACE(our_norm_type, dim_a, 3, ACE_Name=Block_Name + '_ACE_0', status=self.status,
                                 spade_params=[our_norm_type, dim_a, nc_cihp], use_rgb=use_rgb)

            if no_end_norm:
                self.conv = conv(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, norm_layer=Identity,
                                 bias=True)
                # self.conv_norm = SPADE('spadebatch5x5', dim * 2, self.nc_cihp)
                self.conv_norm = ACE(our_norm_type, dim * 2, 3, ACE_Name=Block_Name + '_ACE_0', status=self.status,
                                     spade_params=[our_norm_type, dim * 2, nc_cihp], use_rgb=use_rgb)
            else:
                self.conv = spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1,
                                 bias=use_bias))
                # self.conv_norm = SPADE('spadebatch5x5', dim * 2, self.nc_cihp)
                self.conv_norm = ACE(our_norm_type, dim * 2, 3, ACE_Name=Block_Name + '_ACE_0', status=self.status,
                                     spade_params=[our_norm_type, dim * 2, nc_cihp], use_rgb=use_rgb)

    def forward(self, x, x_seg, style_codes,a=None):
        if a is None:
            # w/o additional input
            residual = x
        else:
            # w/ additional input
            a = self.conv_a(self.activation(self.conv_a_norm(a, x_seg,style_codes)))
            # a = self.conv_a(self.activation(a))
            residual = torch.cat((x, a), dim=1)
        residual = self.conv(self.activation(self.conv_norm(residual, x_seg,style_codes)))
        out = x + residual
        if self.use_dropout:
            out = F.dropout(out, p=0.5, training=self.training)
        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, Block_Name=None, use_rgb=True):
        super().__init__()

        self.use_rgb = use_rgb

        self.Block_Name = Block_Name
        self.status = opt.status

        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        # spade_config_str = opt.norm_G.replace('spectral', '')
        spade_config_str = 'spadebatch5x5'


        ###########  Modifications 1
        normtype_list = ['spadeinstance3x3', 'spadesyncbatch3x3', 'spadebatch3x3']
        our_norm_type = 'spadebatch5x5'

        self.ace_0 = ACE(our_norm_type, fin, 3, ACE_Name= Block_Name + '_ACE_0', status=self.status, spade_params=[spade_config_str, fin, opt.semantic_nc], use_rgb=use_rgb)
        ###########  Modifications 1


        ###########  Modifications 1
        self.ace_1 = ACE(our_norm_type, fmiddle, 3, ACE_Name= Block_Name + '_ACE_1', status=self.status, spade_params=[spade_config_str, fmiddle, opt.semantic_nc], use_rgb=use_rgb)
        ###########  Modifications 1

        if self.learned_shortcut:
            self.ace_s = ACE(our_norm_type, fin, 3, ACE_Name= Block_Name + '_ACE_s', status=self.status, spade_params=[spade_config_str, fin, opt.semantic_nc], use_rgb=use_rgb)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, style_codes, obj_dic=None):


        x_s = self.shortcut(x, seg, style_codes, obj_dic)


        ###########  Modifications 1
        dx = self.ace_0(x, seg, style_codes, obj_dic)

        dx = self.conv_0(self.actvn(dx))

        dx = self.ace_1(dx, seg, style_codes, obj_dic)

        dx = self.conv_1(self.actvn(dx))
        ###########  Modifications 1


        out = x_s + dx
        return out

    def shortcut(self, x, seg, style_codes, obj_dic):
        if self.learned_shortcut:
            x_s = self.ace_s(x, seg, style_codes, obj_dic)
            x_s = self.conv_s(x_s)

        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class DualUnetGenerator_SEAN(nn.Module):
    '''
    Generator with spade, spade in both encoder and decoder.
    '''

    def __init__(self, pose_nc, appearance_nc, output_nc, aux_output_nc=[], nf=32, max_nf=128, num_scales=7,
                 num_warp_scales=5, n_residual_blocks=2, norm='batch', vis_mode='none', activation=nn.ReLU(False),
                 use_dropout=False, no_end_norm=False, gpu_ids=[], use_dismap='',isTrain=True):
        '''
        vis_mode: ['none', 'hard_gate', 'soft_gate', 'residual']
        no_end_norm: remove normalization layer at the start and the end.
        '''
        super(DualUnetGenerator_SEAN, self).__init__()
        self.pose_nc = pose_nc
        self.appearance_nc = appearance_nc
        self.output_nc = output_nc
        self.nf = nf
        self.max_nf = max_nf
        # num of encoder and decoder scales (u-net)
        self.num_scales = num_scales
        self.num_warp_scales = num_warp_scales  # at higher scales, warping will not be applied because the resolution of the feature map is too small
        self.n_residual_blocks = n_residual_blocks
        self.norm = norm
        self.gpu_ids = gpu_ids
        # use dropout in resuial block
        self.use_dropout = use_dropout
        self.vis_mode = vis_mode
        # split person feature and background into different channel
        self.vis_expand_mult = 2  # expanded multiple when perform vis_expand
        self.aux_output_nc = aux_output_nc
        # if first layer of encoder and last layer of decoder to use norm (absolute symmetry for encoder and decoder)
        self.no_end_norm = no_end_norm
        self.is_train = isTrain
        # cihp model's num of labels, here parsing human into 20 part
        self.nc_cihp = 20
        self.usedismap = (use_dismap != '')
        if self.usedismap:
            self.alpha = torch.nn.Parameter(torch.Tensor([-0.1]))
            self.nc_cihp_dec = self.nc_cihp + 12
        else:
            # print('****')
            self.alpha = torch.Tensor([-0.1]).cuda()
            self.nc_cihp_dec = self.nc_cihp

        if norm == 'batch':
            norm_layer = nn.BatchNorm2d
            use_bias = False
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm2d
            use_bias = True
        else:
            raise NotImplementedError()

        #Zcode
        self.Zencoder = Zencoder(3,128)

        ####################################
        # input encoder
        ####################################
        # note maybe here should also has a spade
        if not no_end_norm:
            self.encp_pre_conv = channel_mapping(pose_nc, nf, norm_layer, use_bias)
            self.enca_pre_conv = channel_mapping(appearance_nc, nf, norm_layer, use_bias)
        else:
            self.encp_pre_conv = channel_mapping(pose_nc, nf, Identity, True)
            self.enca_pre_conv = channel_mapping(appearance_nc, nf, Identity, True)
        for l in range(num_scales):
            c_in = min(nf * (l + 1), max_nf)
            c_out = min(nf * (l + 2), max_nf)
            ####################################
            # pose encoder
            ####################################
            # resblocks
            for i in range(n_residual_blocks):
                self.__setattr__('encp_%d_res_%d' % (l, i),
                                 ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=False))
            # down sample
            p_downsample = nn.Sequential(
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            )
            self.__setattr__('encp_%d_downsample' % l, p_downsample)
            ####################################
            # appearance encoder
            ####################################
            for i in range(n_residual_blocks):
                # resblocks
                self.__setattr__('enca_%d_res_%d'%(l, i), ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=False))

                # self.__setattr__('enca_%d_res_%d' % (l, i),
                #                  ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=False,status='train' if self.is_train else 'test'))
                # visibility gating
                if l < num_warp_scales:
                    if vis_mode == 'hard_gate':
                        pass
                    elif vis_mode == 'soft_gate':
                        self.__setattr__('enca_%d_vis_%d' % (l, i),
                                         GateBlock(c_in, c_in * self.vis_expand_mult, activation))
                    elif vis_mode == 'residual':
                        self.__setattr__('enca_%d_vis_%d' % (l, i),
                                         ResidualBlock(c_in, c_in * self.vis_expand_mult, norm_layer, use_bias,
                                                       activation, use_dropout=False))
                    elif vis_mode == 'res_no_vis':
                        self.__setattr__('enca_%d_vis_%d' % (l, i),
                                         ResidualBlock(c_in, None, norm_layer, use_bias, activation, use_dropout=False))
            # down sample
            a_downsample = nn.Sequential(
                activation,
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(c_out)
            )
            self.__setattr__('enca_%d_downsample' % l, a_downsample)
            ####################################
            # decoder
            ####################################
            # resblocks
            if l == num_scales - 1:
                self.dec_fuse = channel_mapping(c_out * 2, c_out, norm_layer,
                                                use_bias)  # a fusion layer at the bottle neck
            # upsample
            upsample_norm = ACE('spadebatch5x5', c_out, 3, ACE_Name='Block' + '_ACE_0', status='train' if self.is_train else 'test',
                                 spade_params=['spadebatch5x5', c_out, self.nc_cihp], use_rgb=True)
            upsample = nn.Sequential(
                activation,
                nn.Conv2d(c_out, c_in * 4, kernel_size=3, padding=1, bias=use_bias),
                nn.PixelShuffle(2),
                norm_layer(c_in)
            )

            self.__setattr__('dec_%d_upsample' % l, upsample)
            self.__setattr__('dec_%d_upsample_norm' % l, upsample_norm)
            for i in range(n_residual_blocks):
                if l == num_scales - 1 and i == n_residual_blocks - 1:
                    self.__setattr__('dec_%d_res_%d' % (l, i),
                                     ResidualBlock_SEAN(c_in, c_in * 2, norm_layer, use_bias, activation, use_dropout,
                                                         no_end_norm=no_end_norm, nc_cihp=self.nc_cihp_dec,status='train' if self.is_train else 'test'))
                else:
                    self.__setattr__('dec_%d_res_%d' % (l, i),
                                     ResidualBlock_SEAN(c_in, c_in * 2, norm_layer, use_bias, activation, use_dropout,
                                                         nc_cihp=self.nc_cihp_dec,status='train' if self.is_train else 'test'))
        ####################################
        # output decoder
        ####################################
        self.dec_output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, output_nc, kernel_size=7, padding=0, bias=True)
        )
        for i, a_nc in enumerate(aux_output_nc):
            dec_aux_output = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(nf, a_nc, kernel_size=7, padding=0, bias=True)
            )
            self.__setattr__('dec_aux_output_%d' % i, dec_aux_output)

    def _vis_expand(self, feat, vis):
        '''
        expand feature from n channels to n*vis_expand_mult channels
        '''
        feat_exp = [feat * (vis == i).float() for i in range(self.vis_expand_mult)]
        return torch.cat(feat_exp, dim=1)

    def forward(self, x_p, x_a, s_seg, d_seg, flow=None, vis=None, dismap=None, output_feats=False,
                single_device=False):
        '''
        x_p: (bsz, pose_nc, h, w), pose input
        x_a: (bsz, appearance_nc, h, w), appearance input
        vis: (bsz, 1, h, w), 0-visible, 1-invisible, 2-background
        flow: (bsz, 2, h, w) or None. if flow==None, feature warping will not be performed
        s_seg: source seg
        d_seg: dest seg
        '''
        if len(self.gpu_ids) > 1 and (not single_device):
            if flow is not None:
                assert vis is not None
                # nn.parallel.data_parallel: not set device_ids, use all gpu. module_kwargs set model args.
                return nn.parallel.data_parallel(self, (x_p, x_a, s_seg, d_seg, flow, vis, dismap),
                                                 module_kwargs={'single_device': True, 'output_feats': output_feats})
            else:
                return nn.parallel.data_parallel(self, (x_p, x_a, s_seg, d_seg),
                                                 module_kwargs={'flow': None, 'vis': None, 'single_device': True,
                                                                'output_feats': output_feats})
        else:

            if dismap is not None:
                dismap = torch.exp(self.alpha * dismap)

            style_codes = self.Zencoder(input=x_a, segmap=s_seg)

            use_fw = flow is not None
            if use_fw:
                vis = vis.round()
            hidden_p = []
            hidden_a = []
            # encoding p
            x_p = self.encp_pre_conv(x_p)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x_p = self.__getattr__('encp_%d_res_%d' % (l, i))(x_p)
                    hidden_p.append(x_p)
                x_p = self.__getattr__('encp_%d_downsample' % l)(x_p)
            # encoding a
            x_a = self.enca_pre_conv(x_a)
            for l in range(self.num_scales):
                for i in range(self.n_residual_blocks):
                    x_a = self.__getattr__('enca_%d_res_%d' % (l, i))(x_a)
                    # feature warping
                    if use_fw and l < self.num_warp_scales:
                        if i == 0:  # compute flow and vis once at each scale
                            flow_l = F.avg_pool2d(flow, kernel_size=2 ** l).div_(2 ** l) if l > 0 else flow
                            vis_l = -F.max_pool2d(-vis,
                                                  kernel_size=2 ** l) if l > 0 else vis  # the priority is visible>invisible>background
                        x_w = warp_acc_flow(x_a, flow_l)
                        if self.vis_mode == 'none':
                            pass
                        # x_w * vis
                        elif self.vis_mode == 'hard_gate':
                            x_w = x_w * (vis_l < 2).float()
                        # x_w * conv(vis)
                        elif self.vis_mode == 'soft_gate':
                            x_we = self._vis_expand(x_w, vis_l)
                            x_w = self.__getattr__('enca_%d_vis_%d' % (l, i))(x_w, x_we)
                        # conv(x_w concat vis)
                        elif self.vis_mode == 'residual':
                            x_we = self._vis_expand(x_w, vis_l)
                            x_w = self.__getattr__('enca_%d_vis_%d' % (l, i))(x_w, x_we)
                        # conv(x_w)
                        elif self.vis_mode == 'res_no_vis':
                            x_w = self.__getattr__('enca_%d_vis_%d' % (l, i))(x_w)
                        hidden_a.append(x_w)
                    else:
                        hidden_a.append(x_a)
                x_a = self.__getattr__('enca_%d_downsample' % l)(x_a)
            # bottleneck fusion
            x = self.dec_fuse(torch.cat((x_p, x_a), dim=1))
            feats = [x]
            # decoding
            if dismap is not None:
                d_seg = torch.cat((d_seg, dismap), 1)
            for l in range(self.num_scales - 1, -1, -1):
                x = self.__getattr__('dec_%d_upsample_norm' % l)(x, d_seg,style_codes)
                x = self.__getattr__('dec_%d_upsample' % l)(x)
                feats = [x] + feats
                for i in range(self.n_residual_blocks - 1, -1, -1):
                    h_p = hidden_p.pop()
                    h_a = hidden_a.pop()
                    x = self.__getattr__('dec_%d_res_%d' % (l, i))(x, d_seg, style_codes,torch.cat((h_p, h_a), dim=1))
            out = self.dec_output(x)
            if self.aux_output_nc or output_feats:
                aux_out = []
                if self.aux_output_nc:
                    for i in range(len(self.aux_output_nc)):
                        aux_out.append(self.__getattr__('dec_aux_output_%d' % i)(x))
                if output_feats:
                    aux_out.append(feats)
                return out, aux_out
            else:
                return out

import os
import pathlib
import torch
import numpy as np
from imageio import imread
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from skimage.morphology import dilation, erosion, square

import glob
import argparse
import pandas as pd
import json
from skimage.draw import circle, polygon
import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lpips
from skimage.measure import compare_ssim


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear')

        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

class FID():
    """docstring for FID
    Calculates the Frechet Inception Distance (FID) to evalulate GANs
    The FID metric calculates the distance between two distributions of images.
    Typically, we have summary statistics (mean & covariance matrix) of one
    of these distributions, while the 2nd distribution is given by a GAN.
    When run as a stand-alone program, it compares the distribution of
    images that are stored as PNG/JPEG at a specified location with a
    distribution given by summary statistics (in pickle format).
    The FID is calculated by assuming that X_1 and X_2 are the activations of
    the pool_3 layer of the inception net for generated samples and real world
    samples respectivly.
    See --help to see further details.
    Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
    of Tensorflow
    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """
    def __init__(self):
        self.dims = 2048
        self.batch_size = 64
        self.cuda = True
        self.verbose=False

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx])
        if self.cuda:
            # TODO: put model into specific GPU
            self.model.cuda()

    def __call__(self, images, gt_path):
        """ images:  list of the generated image. The values must lie between 0 and 1.
            gt_path: the path of the ground truth images.  The values must lie between 0 and 1.
        """
        if not os.path.exists(gt_path):
            raise RuntimeError('Invalid path: %s' % gt_path)


        print('calculate gt_path statistics...')
        m1, s1 = self.compute_statistics_of_path(gt_path, self.verbose)
        print('calculate generated_images statistics...')
        m2, s2 = self.calculate_activation_statistics(images, self.verbose)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


    def calculate_from_disk(self, generated_path, gt_path):
        """
        """
        if not os.path.exists(gt_path):
            raise RuntimeError('Invalid path: %s' % gt_path)
        if not os.path.exists(generated_path):
            raise RuntimeError('Invalid path: %s' % generated_path)

        print('calculate gt_path statistics...')
        m1, s1 = self.compute_statistics_of_path(gt_path, self.verbose)
        print('calculate generated_path statistics...')
        m2, s2 = self.compute_statistics_of_path(generated_path, self.verbose)
        print('calculate frechet distance...')
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)
        print('fid_distance %f' % (fid_value))
        return fid_value


    def compute_statistics_of_path(self, path, verbose):
        npz_file = os.path.join(path, 'statistics.npz')
        if os.path.exists(npz_file):
            f = np.load(npz_file)
            m, s = f['mu'][:], f['sigma'][:]
            f.close()
        else:
            path = pathlib.Path(path)
            files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
            imgs = np.array([imread(str(fn)).astype(np.float32) for fn in files])

            # Bring images to shape (B, 3, H, W)
            imgs = imgs.transpose((0, 3, 1, 2))

            # Rescale images to be between 0 and 1
            imgs /= 255

            m, s = self.calculate_activation_statistics(path, verbose)
            np.savez(npz_file, mu=m, sigma=s)

        return m, s

    def calculate_activation_statistics(self, path, verbose):
        """Calculation of the statistics used by the FID.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the
                         number of calculated batches is reported.
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(path, verbose)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma



    def get_activations(self, path, verbose=False):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : the images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size depends
                         on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the number
                         of calculated batches is reported.
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        self.model.eval()

        path = pathlib.Path(path)
        filenames = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        # filenames = os.listdir(path)
        d0 = len(filenames)

        n_batches = d0 // self.batch_size
        n_used_imgs = n_batches * self.batch_size
        import tqdm
        pred_arr = np.empty((n_used_imgs, self.dims))
        for i in tqdm.tqdm(range(n_batches)):
            # if verbose:
            #     print('\rPropagating batch %d/%d' % (i + 1, n_batches))
                      # end='', flush=True)
            start = i * self.batch_size
            end = start + self.batch_size

            imgs = np.array([imread(str(fn)).astype(np.float32) for fn in filenames[start:end]])

            # Bring images to shape (B, 3, H, W)
            imgs = imgs.transpose((0, 3, 1, 2))

            # Rescale images to be between 0 and 1
            imgs /= 255

            batch = torch.from_numpy(imgs).type(torch.FloatTensor)
            # batch = Variable(batch, volatile=True)
            if self.cuda:
                batch = batch.cuda()

            pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.batch_size, -1)

        if verbose:
            print(' done')

        return pred_arr


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

def get_image_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list'%flist)
    return []

class LPIPS():
    def __init__(self, use_gpu=True):
        self.model = lpips.LPIPS(net='alex').cuda()
        self.use_gpu=use_gpu

    def __call__(self, image_1, image_2):
        """
            image_1: images with size (n, 3, w, h) with value [-1, 1]
            image_2: images with size (n, 3, w, h) with value [-1, 1]
        """
        result = self.model.forward(image_1, image_2)
        return result

    def calculate_from_disk(self, files_1, files_2, batch_size=64, verbose=False):

        result=[]

        d0 = len(files_1)
        assert len(files_1)  == len(files_2)
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0

        n_batches = d0 // batch_size

        for i in tqdm.tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
            start = i * batch_size
            end = start + batch_size
            imgs_1 = np.array(files_1[start:end])
            imgs_2 = np.array(files_2[start:end])
            # Bring images to shape (B, 3, H, W)
            imgs_1 = imgs_1.transpose((0, 3, 1, 2))
            imgs_2 = imgs_2.transpose((0, 3, 1, 2))
            img_1_batch = torch.from_numpy(imgs_1).type(torch.FloatTensor)
            img_2_batch = torch.from_numpy(imgs_2).type(torch.FloatTensor)

            if self.use_gpu:
                img_1_batch = img_1_batch.cuda()
                img_2_batch = img_2_batch.cuda()

            result.append(self.model.forward(img_1_batch, img_2_batch).detach().cpu().numpy())

        distance = np.average(result)
        print('lpips: %.4f'%distance)
        return distance

    def calculate_mask_lpips(self, files_1, files_2, batch_size=64, verbose=False):
        result=[]
        d0 = len(files_1)
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0

        n_batches = d0 // batch_size
        for i in tqdm.tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches))
            start = i * batch_size
            end = start + batch_size
            imgs_1 = np.array(files_1[start:end])
            imgs_2 = np.array(files_2[start:end])
            # Bring images to shape (B, 3, H, W)
            imgs_1 = imgs_1.transpose((0, 3, 1, 2))
            imgs_2 = imgs_2.transpose((0, 3, 1, 2))

            img_1_batch = torch.from_numpy(imgs_1).type(torch.FloatTensor)
            img_2_batch = torch.from_numpy(imgs_2).type(torch.FloatTensor)

            if self.use_gpu:
                img_1_batch = img_1_batch.cuda()
                img_2_batch = img_2_batch.cuda()

            result.append(self.model.forward(img_1_batch, img_2_batch).detach().cpu().numpy())

        distance = np.average(result)
        print('masked lpips: %.4f'%distance)
        return distance

def produce_ma_mask(kp_array, img_size=(128, 64), point_radius=4):
    MISSING_VALUE = -1
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
              [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
               [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    limbs = np.array(limbs) - 1
    for f, t in limbs:
        from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
        to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue

        norm_vec = kp_array[f] - kp_array[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)
        vetexes = np.array([
            kp_array[f] + norm_vec,
            kp_array[f] - norm_vec,
            kp_array[t] - norm_vec,
            kp_array[t] + norm_vec
        ])
        yy, xx = polygon(vetexes[:, 0], vetexes[:, 1], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(kp_array):
        if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=point_radius, shape=img_size)
        mask[yy, xx] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))
    return mask

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def masked_ssim_score(generated_images, reference_images):
    ssim_score_list = []
    for reference_image, generated_image in tqdm.tqdm(zip(reference_images, generated_images)):
        ssim = compare_ssim(reference_image, generated_image,gaussian_weights=True, sigma=1.5,
                            use_sample_covariance=False, multichannel=True,
                            data_range=generated_image.max() - generated_image.min())
        ssim_score_list.append(ssim)
    print ("masked SSIM %.3f" % np.mean(ssim_score_list))
    return np.mean(ssim_score_list)

def load_generated_images(generated, gt, annotation_file):
    target_images,generated_images,target_images_m,generated_images_m, stm, sgm = [],[],[],[],[],[]
    df = pd.read_csv(annotation_file, sep=':')
    for file in tqdm.tqdm(os.listdir(generated)):
        if not file.endswith('.jpg'):
            continue
        gntimg = imread(os.path.join(generated,file))
        gtimg= imread(os.path.join(gt,file))
        fs = file.split('_')
        name=fs[4]+'_'+fs[5]+'_'+fs[6]+'_'+fs[7]
        ano_to = df[df['name'] == name].iloc[0]
        kp_to = load_pose_cords_from_strings(ano_to['keypoints_y'], ano_to['keypoints_x'])

        mask = produce_ma_mask(kp_to, (128,64))

        stm.append(gtimg * mask[..., np.newaxis])
        sgm.append(gntimg * mask[..., np.newaxis])
        generated_images.append(gntimg.astype(np.float32) / 127.5 - 1)
        target_images.append(gtimg.astype(np.float32) / 127.5 - 1)
        generated_images_m.append((gntimg.astype(np.float32) / 127.5 - 1) * mask[..., np.newaxis])
        target_images_m.append((gtimg.astype(np.float32) / 127.5 - 1) * mask[..., np.newaxis])

    return generated_images,generated_images_m,target_images,target_images_m, stm, sgm


if __name__ == "__main__":
    print('load start')

    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--gt_path', help='Path to ground truth data', type=str)
    parser.add_argument('--distorated_path', help='Path to output data', type=str)
    parser.add_argument('--fid_real_path', help='Path to real images when calculate FID', type=str)
    parser.add_argument('--bonesLst',default='datasets/market1501/label/market-annotation-test.csv',help='Path to annotation',type=str)
    args = parser.parse_args()

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    lpips = LPIPS()
    print('load LPIPS')

    fid = FID()
    print('load FID')

    print('calculate fid metric...')
    fid_score = fid.calculate_from_disk(args.distorated_path, args.fid_real_path)

    print('load imgs...')
    generated_images,generated_images_m,target_images,target_images_m, stm, sgm = load_generated_images(args.distorated_path,args.gt_path, args.bonesLst)

    print('calculate lpips metric...')
    lpips_score = lpips.calculate_from_disk(generated_images, target_images)

    print('calculate masked lpips and SSIM metric...')
    lpips_masked = lpips.calculate_mask_lpips(generated_images_m,target_images_m)
    structured_masked = masked_ssim_score(sgm, stm)

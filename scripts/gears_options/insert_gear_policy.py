import numpy as np
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import ResNet50Layers, ResNet101Layers, ResNet152Layers

from chainer import iterators
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainer.datasets import split_dataset
from chainer import optimizers
from chainer import serializers

from tqdm import tqdm
import math

import chainer
import chainer.functions as F
import chainer.distributions as D
import chainer.links as L
import numpy as np

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils
from mdn import MDN

IMAGE_SIZE=128


class Encoder(chainer.Chain):

    def __init__(self, n_first, n_latent):
        super(Encoder, self).__init__()

        self.n_first = n_first
        self.n_latent = n_latent
        with self.init_scope():
            n_ch = 3 # RGB
            self.e_c0 = L.Convolution2D(in_channels=None, out_channels=n_first, ksize=4, stride=2, pad=1)
            self.e_c1 = L.Convolution2D(n_first, n_first * 2, 4, 2, 1)
            self.e_c2 = L.Convolution2D(n_first * 2, n_first * 4, 4, 2, 1)
            self.e_c3 = L.Convolution2D(n_first * 4, n_first * 8, 4, 2, 1)
            self.e_c4 = L.Convolution2D(n_first * 8, n_first * 16, 4, 2, 1)

            self.e_bn1 = L.BatchNormalization(n_first * 2, use_gamma=False)
            self.e_bn2 = L.BatchNormalization(n_first * 4, use_gamma=False)
            self.e_bn3 = L.BatchNormalization(n_first * 8, use_gamma=False)
            self.e_bn4 = L.BatchNormalization(n_first * 16, use_gamma=False)

            self.e_mu = L.Linear(n_first * 8 * 8 * 4, n_latent)
            self.e_ln_var = L.Linear(n_first * 8 * 8 * 4, n_latent)

    def forward(self, x):
        # h = F.tanh(self.linear(x))
        # mu = self.mu(h)
        # ln_sigma = self.ln_sigma(h)  # log(sigma)
        # return D.Normal(loc=mu, log_scale=ln_sigma)

        mu, ln_var = self.forward_dist(x)

        z = F.gaussian(mu, ln_var)
        return z

    def forward_dist(self, x):
        # print('forward_dist', x.shape)
        h = F.leaky_relu(self.e_c0(x), slope=0.2)
        # print('1 ', h.shape)
        h = F.leaky_relu(self.e_bn1(self.e_c1(h)), slope=0.2)
        # print('2 ', h.shape)
        h = F.leaky_relu(self.e_bn2(self.e_c2(h)), slope=0.2)
        # print('3 ', h.shape)
        h = F.leaky_relu(self.e_bn3(self.e_c3(h)), slope=0.2)
        # print('4 ', h.shape)
        h = F.leaky_relu(self.e_bn4(self.e_c4(h)), slope=0.2)
        # print('5 ', h.shape)
        h = F.reshape(h, (x.shape[0], -1)) # self.n_first * 8 * 4
        # print('6 ', h.shape)
        mu = self.e_mu(h)
        ln_var = self.e_ln_var(h)
        # print('mu ', mu.shape)

        return mu, ln_var



class Decoder(chainer.Chain):

    def __init__(self, n_first, n_latent):
        super(Decoder, self).__init__()
        self.n_first = n_first
        with self.init_scope():
            self.d_l0 = L.Linear(n_latent, n_first * 16 * 4)
            self.d_dc0 = L.Deconvolution2D(in_channels=n_first * 16, out_channels=n_first * 16, ksize=4, stride=2, pad=1)
            self.d_dc1 = L.Deconvolution2D(in_channels=n_first * 16, out_channels=n_first * 8, ksize=4, stride=2, pad=1)
            self.d_dc2 = L.Deconvolution2D(in_channels=n_first * 8, out_channels=n_first * 4, ksize=4, stride=2, pad=1)
            self.d_dc3 = L.Deconvolution2D(in_channels=n_first * 4, out_channels=n_first * 2, ksize=4, stride=2, pad=1)
            self.d_dc4 = L.Deconvolution2D(in_channels=n_first * 2, out_channels=n_first, ksize=4, stride=2, pad=1)
            self.d_dc5 = L.Deconvolution2D(in_channels=n_first, out_channels=3, ksize=4, stride=2, pad=1)
            self.d_bn0 = L.BatchNormalization(n_first * 16, use_gamma=False)
            self.d_bn1 = L.BatchNormalization(n_first * 8, use_gamma=False)
            self.d_bn2 = L.BatchNormalization(n_first * 4, use_gamma=False)
            self.d_bn3 = L.BatchNormalization(n_first * 2, use_gamma=False)
            self.d_bn4 = L.BatchNormalization(n_first, use_gamma=False)
            self.d_bn5 = L.BatchNormalization(3, use_gamma=False)

    def forward(self, z, sigmoid=True):
        # print(z)
        # print(self.d_l0(z))
        h = F.relu(self.d_l0(z))
        # print h.shape
        h = F.reshape(h, (-1, self.n_first * 16, 2, 2))
        # print h.shape
        h = F.relu(self.d_bn0(self.d_dc0(h)))
        # print h.shape
        h = F.relu(self.d_bn1(self.d_dc1(h)))
        # print h.shape
        h = F.relu(self.d_bn2(self.d_dc2(h)))
        # print h.shape
        h = F.relu(self.d_bn3(self.d_dc3(h)))
        # print h.shape
        h = F.relu(self.d_bn4(self.d_dc4(h)))
        h = self.d_bn5(self.d_dc5(h))

                # need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
           return F.sigmoid(h)
        else:
           return h


from chainer import backend
# import cupy as cp

class SpatialDecoder(chainer.Chain):
    """docstring for SpatialDecoder"""
    def __init__(self, n_first, n_latent):
        super(SpatialDecoder, self).__init__()
        self.n_first = n_first
        self.n_latent = n_latent

        with self.init_scope():
            self.sp_dec_3 = L.Convolution2D(self.n_latent+2, 64, ksize=3, pad=1)
            self.sp_dec_2 = L.Convolution2D(64, 64, ksize=3, pad=1)
            self.sp_dec_1 = L.Convolution2D(64, 64, ksize=3, pad=1)
            self.sp_dec_0 = L.Convolution2D(64, 3, ksize=3, pad=1)
        
    def forward(self, latent, sigmoid=True):
        image_size = self.n_first
        xp = backend.get_array_module(latent)

        a = xp.linspace(-1, 1, image_size)
        b = xp.linspace(-1, 1, image_size)

        x, y = xp.meshgrid(a, b)

        x = x.reshape(image_size, image_size, 1)
        y = y.reshape(image_size, image_size, 1)

        xy = xp.concatenate((x,y), axis=-1)

        batchsize = len(latent)
        xy_tiled = xp.tile(xy, (batchsize, 1, 1, 1)).astype(xp.float32)

        latent_tiled = F.tile(latent, (1, 1, image_size*image_size)).reshape(batchsize, image_size, image_size, self.n_latent)
        latent_and_xy = F.concat((latent_tiled, xy_tiled), axis=-1)
        latent_and_xy = F.swapaxes(latent_and_xy, 1, 3)

        sp_3_decoded = F.relu(self.sp_dec_3(latent_and_xy))
        sp_2_decoded = F.relu(self.sp_dec_2(sp_3_decoded))
        sp_1_decoded = F.relu(self.sp_dec_1(sp_2_decoded))
        out_img = self.sp_dec_0(sp_1_decoded)

        # need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
           return F.sigmoid(out_img)
        else:
           return out_img


class FCNAction(chainer.Chain):
    """Fully Connected Action network"""
    def __init__(self, n_in, n_out, n_hidden=100):
        super(FCNAction, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden

        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_out)
        
    def forward(self, x):

        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        out = self.l3(h)

        return out


class InsertGearPolicy(chainer.Chain):
    """docstring for InsertGearPolicyResNet"""
    def __init__(self, action_output_space = 7):
        super(InsertGearPolicy, self).__init__()
        
        with self.init_scope():
            n_latent = 512
            self.encode_model = Encoder(n_first=16, n_latent=n_latent)
            self.encode_model.layer_size = n_latent

            self.action_output_space = action_output_space

            self.action_n_in = self.encode_model.layer_size
            self.action_n_out = self.action_output_space
            self.action_n_hidden = 100
            self.action = FCNAction(self.action_n_in, self.action_n_out, self.action_n_hidden)

            self.decoder_model = Decoder(n_first=IMAGE_SIZE, n_latent=int(self.encode_model.layer_size))
            # self.decoder_model = SpatialDecoder(n_first=IMAGE_SIZE, n_latent=int(self.encode_model.layer_size/2.))

    def forward(self, x):
        # print(x.shape)
        h = self.encode_model.forward(x)
        h2 = self.action.forward(h)

        return h2

    def forward_with_z(self, x):
        z_mu, z_ln_var = self.encode_model.forward_dist(x)

        z = F.gaussian(z_mu, z_ln_var)

        h = self.action.forward(z)

        return h, z_mu, z_ln_var

    def vae_image(self, x):
        z = self.encode_model.forward(x)
        # print(z.shape, x.shape)
        return self.decoder_model(z, sigmoid=True)

    def VAE_loss_func(self, C=1.0, k=1, train=True):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
            train (bool): If true loss_function is used for training.
        """

        def lf(self, x, mu, ln_var, split=False):
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decoder_model(z, sigmoid=False)) / (k * batchsize)
            rec_loss = rec_loss
            kl_loss = C * F.gaussian_kl_divergence(mu, ln_var) / batchsize
            loss = rec_loss +  kl_loss
            if split:
                return rec_loss, kl_loss
            else:
                return loss
        return lf

    def calc_loss(self, x, t):
        # h = self.encode_model.feature(x)
        # print('encode_model_space', h)
        xp = backend.get_array_module(x)

        VAE_LOSS_SCALE = 1e-5
        JOINT_LOSS_WEIGHTS = xp.linspace(10, 1, t.shape[1])
        # print(JOINT_LOSS_WEIGHTS)

        # action model output
        output, z_mu, z_ln_var = self.forward_with_z(x)
        # print(t.shape, output.shape, z_mu.shape, z_ln_var.shape)

        # MAR of action model
        self.mean_abs_error = F.mean_absolute_error(t, output)
        self.weighted_joint_error = F.mean(F.squared_error(t, output) * JOINT_LOSS_WEIGHTS)

        # VAE loss
        self.vae_loss_rec, self.vae_loss_kl = self.VAE_loss_func()(self, x, z_mu, z_ln_var, split=True)
        self.vae_loss = VAE_LOSS_SCALE * (self.vae_loss_rec + self.vae_loss_kl)
        
        # Total loss
        self.total_loss = self.weighted_joint_error + \
                          self.vae_loss + self.action.get_loss(output, t) ## TODO: FIX this

        chainer.report({'mae': self.mean_abs_error}, self)
        chainer.report({'weighted': self.weighted_joint_error}, self)
        chainer.report({'VAE': self.vae_loss}, self)
        chainer.report({'VAE_KL': self.vae_loss_kl}, self)
        chainer.report({'VAE_REC': self.vae_loss_rec}, self)
        chainer.report({'loss': self.total_loss}, self)
        return self.total_loss

    def load_model(self, filename='my.model'):
        serializers.load_npz(filename, self)
        print('Loaded `{}` model.'.format(filename))

    def prepare(self, raw):
        x, y, width, height = 200, 120, 590, 460 
        raw = raw[y:y + height, x:x + width]

        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE)).transpose((2, 0, 1))
        raw = raw/255.

        return raw.astype(np.float32)

####

class InsertGearPolicyResNet(chainer.Chain):
    """docstring for InsertGearPolicyResNet"""
    def __init__(self, action_output_space=7):
        super(InsertGearPolicyResNet, self).__init__()
        
        with self.init_scope():
            self.encode_model = ResNet50Layers()
            self.encode_model.pick = 'pool5'
            self.encode_model.layer_size = 2048
            self.encode_model.feature = \
                    lambda x: self.encode_model.forward(x, 
                              layers=[self.encode_model.pick])[self.encode_model.pick]

            self.action_output_space = action_output_space

            # self.action_n_in = self.encode_model.layer_size
            # self.action_n_out = self.action_output_space
            # self.action_n_hidden = 100
            # self.action = FCNAction(self.action_n_in, self.action_n_out, self.action_n_hidden)

            self.mdn_hidden_units = 100
            self.mdn_gaussian_mixtures = 20
            self.mdn_input_dim = self.action_output_space
            self.action = MDN(self.mdn_input_dim, self.mdn_hidden_units, self.mdn_gaussian_mixtures)


            self.decoder_model = Decoder(n_first=IMAGE_SIZE, n_latent=int(self.encode_model.layer_size/2.))
            # self.decoder_model = SpatialDecoder(n_first=IMAGE_SIZE, n_latent=int(self.encode_model.layer_size/2.))

    def forward(self, x):
        # print(x.shape)
        h = self.encode_model.feature(x)
        h2 = self.action.forward(h)

        return h2

    def forward_with_z(self, x):
        h = self.encode_model.feature(x)
        
        l = h.shape[1]
        z_mu = h[:, :l/2]
        z_ln_var = h[:, l/2:]

        h2 = self.action.forward(h)

        return h2, z_mu, z_ln_var

    def forward_with_output_z(self, x):
        h = self.encode_model.feature(x)

        a_mu, a_ln_var = self.action.sample_distribution(h)

        return a_mu, a_ln_var

    def vae_image(self, x):
        h = self.encode_model.feature(x)
        l = h.shape[1]
        z_mu = h[:, :l/2]
        z_ln_var = h[:, l/2:]
        return self.decoder_model(F.gaussian(z_mu, z_ln_var), sigmoid=True)

    JOINT_SCALES_MIN=np.asarray([-3.2901096, -11.005876, -4.4152613, -3.6181638, -5.951669, -2.7354536, -3.1384876]) 
    JOINT_SCALES_MAX=np.asarray([6.1805234, 9.23313, 8.043016, 3.5986853, 6.849108, 6.883955, 2.8147655])
    JOINT_SCALES_RANGE=np.abs(JOINT_SCALES_MIN - JOINT_SCALES_MAX)
    
    def prepare_joints(self, raw):
        print(raw.shape)
        raw -= self.JOINT_SCALES_MIN
        raw /= self.JOINT_SCALES_RANGE
        return raw

    def make_real_joints(self, raw):
        raw *= self.JOINT_SCALES_RANGE
        raw += self.JOINT_SCALES_MIN 
        return raw

    #### DUPLICATE below
    def VAE_loss_func(self, C=1.0, k=1, train=True):
        """Get loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.
        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
            train (bool): If true loss_function is used for training.
        """

        def lf(self, x, mu, ln_var, split=False):
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decoder_model(z, sigmoid=False)) / (k * batchsize)
            rec_loss = rec_loss
            kl_loss = C * F.gaussian_kl_divergence(mu, ln_var) / batchsize
            loss = rec_loss +  kl_loss
            if split:
                return rec_loss, kl_loss
            else:
                return loss
        return lf

    def calc_loss(self, x, t):
        # h = self.encode_model.feature(x)
        # print('encode_model_space', h)
        xp = backend.get_array_module(x)

        VAE_LOSS_SCALE = 1e-5
        JOINT_LOSS_WEIGHTS = xp.linspace(10, 1, t.shape[1])
        # print(JOINT_LOSS_WEIGHTS)

        # action model output
        output, z_mu, z_ln_var = self.forward_with_z(x)
        # print(t.shape, output.shape, z_mu.shape, z_ln_var.shape)

        # MAR of action model
        self.mean_abs_error = F.mean_absolute_error(t, output)
        self.weighted_joint_error = F.mean(F.squared_error(t, output) * JOINT_LOSS_WEIGHTS)
        self.gnll = self.action.negative_log_likelihood(F.concat((z_mu, z_ln_var)), t)

        # VAE loss
        self.vae_loss_rec, self.vae_loss_kl = self.VAE_loss_func()(self, x, z_mu, z_ln_var, split=True)
        self.vae_loss = VAE_LOSS_SCALE * (self.vae_loss_rec + self.vae_loss_kl)
        
        # Total loss
        self.total_loss = self.weighted_joint_error + \
                          self.vae_loss + self.gnll

        chainer.report({'mae': self.mean_abs_error}, self)
        chainer.report({'gnll': self.gnll}, self)
        chainer.report({'weighted': self.weighted_joint_error}, self)
        chainer.report({'VAE': self.vae_loss}, self)
        chainer.report({'VAE_KL': self.vae_loss_kl}, self)
        chainer.report({'VAE_REC': self.vae_loss_rec}, self)
        chainer.report({'loss': self.total_loss}, self)
        return self.total_loss

    def load_model(self, filename='my.model'):
        serializers.load_npz(filename, self)
        print('Loaded `{}` model.'.format(filename))

    def prepare(self, raw):

        # raw = cv2.GaussianBlur(raw,(0, 0), sigmaX=2)
        
        x, y, width, height = 200, 120, 590, 460 
        raw = raw[y:y + height, x:x + width]

        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE)).transpose((2, 0, 1))
        raw = raw/255.

        return raw.astype(np.float32)


        #### DUPLICATE



####

import cv2
import glob


def load_data(prep_f, prepare_joints, subset, skipcount):
    joints = load_joints(skip_first=True, subset=subset)
    frames = load_frames(skip_last=True, prep_f=prep_f, subset=subset)

    if prepare_joints:
        joints = prepare_joints(joints)

    if skipcount > 1:
        joints = joints[::skipcount]
        frames = frames[::skipcount]

    # Realign data to corespond image(t) -> joint_state(t+1)
    # This is now done in the load_joints and load_frames methods.
    # remove the first joints data
    # joints.pop(0)
    # remove last image frame
    # frames.pop(-1)

    print(len(joints), len(frames))
    assert(len(joints) == len(frames) and len(joints) and len(frames))

    return np.asarray(frames).astype(np.float32), np.asarray(joints).astype(np.float32)


def load_joints(filename=None, min_joint=31, max_joint=38, verbose=0, skip_first=True, subset=-1):
    folders = sorted(glob.glob('data/demo_*'))

    if subset >= 0:
        folders = folders[:subset]

    joints = []
    for folder in tqdm(folders):
        files = sorted(glob.glob(folder + '/joint_position_*'))
        # print(len(files))
        skipped = False

        if verbose > 0:
            print(files)

        for f in files:
            joint = np.loadtxt(f)
            ret = joint is not None

            if ret:
                if verbose > 0:
                    print(joint.shape)

                if skip_first and not skipped:
                    skipped = True
                else:
                    joints.append(np.asarray(joint).astype(np.float32)[min_joint:max_joint])
            else:
                break

    # When everything done, release the capture
    if len(joints) > 0:
        print('Done loading joints data. ({}).'.format(len(joints)))
    else:
        print('Didn\'t load any joints. Please check file ', filename, ' exists.')

    return np.asarray(joints)

def load_frames(filename=None, size=(IMAGE_SIZE, IMAGE_SIZE), data_size=100, verbose=0, skip_last=True, prep_f=None, subset=-1):
    folders = sorted(glob.glob('data/demo_*'))

    if subset >= 0:
        folders = folders[:subset]


    frames = []
    for folder in tqdm(folders):
        files = sorted(glob.glob(folder + '/kinect2_qhd_image_color_rect_*.jpg'))
        # files = sorted(glob.glob(folder + '/kinect2_qhd_image_depth_rect_*.jpg'))
        # print(len(files))

        if verbose > 0:
            # print(files)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('frame', 800,800)

        for f in files:
            frame = cv2.imread(f)
            ret = frame is not None

            if ret:
                if verbose > 0:
                    print(frame.shape)
                    cv2.imshow('frame',frame)
                # frame = cv2.resize(frame, size)
                # frame = frame.astype(np.float32)/255.
                # frame = cv2.resize(frame, size).transpose((2, 1, 0))

                # frame = chainer.links.model.vision.resnet.prepare(frame, size)
                if prep_f:
                    frame = prep_f(frame)
                else:
                # x, y, width, height = 200, 120, 590, 460 
                # frame = frame[y:y + height, x:x + width]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, size).transpose((2, 0, 1))
                    frame = frame/255.

                frames.append(frame)
            else:
                break
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        if skip_last and len(files) > 0:
            frames.pop(-1)

    # When everything done, release the capture
    if verbose > 0:
        cv2.destroyAllWindows()
    if len(frames) > 0:
        print('Done loading images data. ({}).'.format(len(frames)))
    else:
        print('Didn\'t load any frames. Please check file ', filename, ' exists.')

    return frames


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def colour_agumentations(images, joints, n):
    ''' n - how many times to expand the dataset with augmentaitons
    '''
    all_images = []
    all_joints = []
    import utils_image as ui

    for i, j in zip(images, joints):
        # Add original
        all_images.append(i)
        all_joints.append(j)

        # Add extra
        for _ in range(n-1):
            i_d = ui.random_distort(i*255)/255
            
            all_images.append(i_d)
            all_joints.append(j)

    return np.asarray(all_images), np.asarray(all_joints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', '-g', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=30)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--real_test',  dest='real_test', action='store_true', 
                                help='Whether to split the data or use a complete new trial.')
    parser.add_argument('--aug', type=int, default=2,
                                help='How many times to increase the dataset with augmented images.')
    parser.add_argument('--subset', type=int, default=-1, help='Should we read just `x` number of folders.')
    parser.add_argument('--skipcount', type=int, default=1, help='Take every `x`-th frame from a sequence.')
    # parser.add_argument('--blackout',  dest='blackout', action='store_true', 
    #                             help='Whether to blackout part of the image or not.')
    # parser.add_argument('--mdn_hidden-units', '-u', type=int, default=24)
    # parser.add_argument('--mdn_gaussian-mixtures', '-m', type=int, default=24)
    parser.add_argument('--max_epoch', '-e', type=int, default=2500)
    parser.add_argument('--resume', '-r', type=int, default=None)
    parser.add_argument('--out_dir', '-o', type=str, default='results/result_test')
    # parser.add_argument('--data_base_dir', type=str, default='/media/daniel/data/hhc/')
    # parser.add_argument('--data_file_pattern', '-f', type=str, default='trial{}.avi')
    args = parser.parse_args()


    model = InsertGearPolicyResNet()
    # frames, joints = load_data(prep_f=model.prepare, prepare_joints=model.prepare_joints) # Scale it all
    frames, joints = load_data(prep_f=model.prepare, prepare_joints=None, 
                               subset=args.subset, skipcount=args.skipcount) # Scale only images
    print('Frames shape: ', frames.shape, ' joints shape: ', joints.shape)

    from sklearn.model_selection import train_test_split
    frames_train, frames_test, joints_train, joints_test = train_test_split(
                                    frames, joints, test_size=args.test_split, random_state=42)

    if args.aug > 1:
        frames_train, joints_train = colour_agumentations(frames_train, joints_train, n=args.aug)
        print('After augmentation. Frames shape: ', frames.shape, ' joints shape: ', joints.shape)

    data_train = chainer.datasets.TupleDataset(frames_train, joints_train)
    data_test = chainer.datasets.TupleDataset(frames_test, joints_test)
    train_iter = iterators.SerialIterator(data_train, args.batch_size, shuffle=True)
    test_iter = iterators.SerialIterator(data_test, args.batch_size, repeat=False, shuffle=False)


    # frames, joints = unison_shuffled_copies(frames, joints)

    # data = chainer.datasets.TupleDataset(frames, joints)
    # print('Dataset length: ', data._length)

    # print('Frame size: ', data[0][0].shape, data[0][0].dtype)

    # if args.real_test:
    #     print('Using test trial.')
    #     train_iter = iterators.SerialIterator(data, args.batch_size, shuffle=True)

    #     # Load the test data
    #     print('Not done.')
    #     exit(0)
    #     # test_frames, test_joints = load_frames_labels(ids=[11], filestype=''.join((args.data_base_dir, args.data_file_pattern)), blackout=args.blackout)
    #     data_test = chainer.datasets.TupleDataset(test_frames, test_joints)
    #     test_iter = iterators.SerialIterator(data_test, args.batch_size, repeat=False, shuffle=False)
    # else:
    #     print('Splitting data at ratio: ', args.test_split)
    #     data_test, data_train = split_dataset(data, int(args.test_split*len(data)))
    #     train_iter = iterators.SerialIterator(data_train, args.batch_size, shuffle=True)
    #     test_iter = iterators.SerialIterator(data_test, args.batch_size, repeat=False, shuffle=False)


    if args.gpu_id >= 0:
        print('Loading model to gpu', args.gpu_id)
        chainer.backends.cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu(args.gpu_id)


    # Create the optimizer for the model
    optimizer = optimizers.Adam().setup(model)
    # optimizer = optimizers.SGD().setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=1e-6))
    # optimizer.add_hook(chainer.optimizer_hooks.GradientHardClipping(-.1, .1))


    # xp = chainer.backend.get_array_module(data_train)
    # optimizer.update(model.calc_loss, xp.asarray([data_train[0][0]]), xp.asarray([data_train[0][1]]))
    # import chainer.computational_graph as c
    # g = c.build_computational_graph(model.calc_loss)
    # with open('results/graph.dot', 'w') as o:
    #     o.write(g.dump())

    updater = training.StandardUpdater(train_iter, optimizer, 
                                       loss_func=model.calc_loss,
                                       device=args.gpu_id)
    # Resume from a specified snapshot
    if args.resume:
        print('Loading from resume snapshot: ', args.resume, '{}/snapshot_epoch_{}.trainer'.format(args.out_dir, args.resume))
        chainer.serializers.load_npz('{}/snapshot_epoch_{}.trainer'.format(args.out_dir, args.resume), trainer)

    # Pre-training
    # print('Pretraining started.')
    # trainer = training.Trainer(updater, (3, 'epoch'), out=args.out_dir)
    
    

    # # Disable update for the head model
    # print('Disabling training of head model.')
    # model.encode_model.disable_update()
    # trainer.extend(extensions.ProgressBar())
    # trainer.extend(extensions.FailOnNonNumber())
    # trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/mae', 'main/VAE', 'validation/main/loss', 'validation/main/mae', 'validation/main/VAE', 'elapsed_time']))
    # trainer.extend(utils.display_image(model.vae_image, data_test, args.out_dir, args.gpu_id), trigger=(1, 'epoch'))
    # trainer.run()

    # Full training
    print('Full model training ...')
    trainer = training.Trainer(updater, (args.max_epoch, 'epoch'), out=args.out_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, eval_func=model.calc_loss, device=args.gpu_id), name='val', trigger=(1, 'epoch'))
    # trainer.extend(extensions.Evaluator(test_iter, {'m':model}, eval_func=model.calc_loss, device=args.gpu_id), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/mae', 'main/gnll', 'main/weighted', 'main/VAE', 'main/VAE_REC','main/VAE_KL', 'val/main/loss', 'val/main/mae', 'val/main/weighted', 'elapsed_time']))#, 'val/main/VAE', 'main/loss', 'validation/main/loss', 'elapsed_time'], ))
    trainer.extend(extensions.PlotReport(['main/mae', 'val/main/mae', 'main/VAE', 'val/main/VAE'], x_key='epoch', file_name='loss.png', marker=None))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.FailOnNonNumber())
    # Save every X epochs
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.trainer'), trigger=(200, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, '%s_model_epoch_{.updater.epoch}.model' % (model.__class__.__name__)), trigger=(10, 'epoch'))
    # # Take a best snapshot
    # record_trigger = training.triggers.MinValueTrigger('validation/main/mae', (1, 'epoch'))
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=record_trigger)
    # trainer.extend(extensions.snapshot_object(model, '%s_best_model.npz' % (model.__class__.__name__)), trigger=record_trigger)

    trainer.extend(utils.display_image(model.vae_image, data_test, args.out_dir, args.gpu_id, n=3), trigger=(1, 'epoch'))
# FOR SGD   trainer.extend(extensions.ExponentialShift('lr', 0.5, init=1e-4, target=1e-8), trigger=(200, 'epoch'))
    trainer.extend(extensions.ExponentialShift('alpha', 0.5, init=1e-3, target=1e-8), trigger=(100, 'epoch'))


    # Disable/Enable update for the head model
    model.encode_model.enable_update()


    trainer.run()
    print('Done.')

if __name__ == '__main__':
    main()

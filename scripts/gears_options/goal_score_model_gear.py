import numpy as np
import argparse, glob, cv2

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import ResNet50Layers

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

import insert_gear_policy as igp 

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils
from mdn import MDN

IMAGE_SIZE=128

class GoalScoreModel(igp.InsertGearPolicyResNet):
    """docstring for GoalScoreModel"""
    def __init__(self):
        super(GoalScoreModel, self).__init__(action_output_space = 1)
        

    def prepare(self, raw):
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE)).transpose((2, 0, 1))
        raw = raw/255.

        return raw.astype(np.float32)
        

####

class GoalScoreJointModel(chainer.Chain):
    """docstring for GoalScoreJointModel"""
    def __init__(self):
        super(GoalScoreJointModel, self).__init__()
        with self.init_scope():
            self.action_output_space = 1

            self.mdn_hidden_units = 100
            self.mdn_gaussian_mixtures = 20
            self.mdn_input_dim = self.action_output_space
            self.action = MDN(self.mdn_input_dim, self.mdn_hidden_units, self.mdn_gaussian_mixtures)
        
    def forward(self, x):
        out = self.action(x)
        return out

    def calc_loss(self, x, t):
        self.loss = self.action.get_loss(x, t)
        return loss


class DynamicsStateModel(chainer.Chain):
    """docstring for DynamicsStateModel"""
    def __init__(self):
        super(DynamicsStateModel, self).__init__()
        with self.init_scope():
            self.action_output_space = 2048

            self.mdn_hidden_units = 100
            self.mdn_gaussian_mixtures = 20
            self.mdn_input_dim = self.action_output_space
            self.fwd_step = MDN(self.mdn_input_dim, self.mdn_hidden_units, self.mdn_gaussian_mixtures)
        
    def forward(self, x):
        out = self.fwd_step(x)
        return out

    def calc_loss(self, x, t):
        self.loss = self.fwd_step.get_loss(x, t)
        return loss


def load_frames(folder_id=2, size=(IMAGE_SIZE, IMAGE_SIZE), image_style='/kinect2_qhd_image_color_rect_*.jpg', data_size=100, verbose=0, skip_last=True, prep_f=None, subset=-1):
    '''
    This is a tweaked version of the igp.load_frames in order to return the scale for goal score.
    '''
    folders = sorted(glob.glob('/mnt/7ac4c5b9-8c05-451f-9e6d-897daecb7442/gears/full_demos/demo{}/demo_*'.format(folder_id)))

    if subset >= 0:
        folders = folders[:subset]


    frames = []
    goal_scores = []
    for folder in tqdm(folders):
        files = sorted(glob.glob(folder + image_style))
        # files = sorted(glob.glob(folder + '/kinect2_qhd_image_color_rect_*.jpg'))
        # files = sorted(glob.glob(folder + '/r_forearm_cam_image_rect_color_*.jpg'))
        # files = sorted(glob.glob(folder + '/l_forearm_cam_image_rect_color_*.jpg'))
        # files = sorted(glob.glob(folder + '/kinect2_qhd_image_depth_rect_*.jpg'))

        if verbose > 0:
            print(files)
            print(len(files))
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        # Add the lables
        goal_scores += list(np.linspace(0, 1, len(files)).astype(np.float32).reshape(-1, 1))

        # Add the images
        for f in files:
            frame = cv2.imread(f)
            ret = frame is not None

            if ret:
                if verbose > 0:
                    print(frame.shape)
                    cv2.imshow('frame',frame)

                if prep_f:
                    frame = prep_f(frame)

                frames.append(frame)
            else:
                break
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    if verbose > 0:
        cv2.destroyAllWindows()
    if len(frames) > 0:
        print('Done loading images data. ({}).'.format(len(frames)))
    else:
        print('Didn\'t load any frames. Please check file ', filename, ' exists.')

    return np.asarray(frames, dtype=np.float32), np.asarray(goal_scores, dtype=np.float32)

def load_all_data(prep_f):
    all_frames = []
    all_labels = []
    for i in tqdm(range(1, 6)):
        frames, labels = load_frames(folder_id=i, prep_f=prep_f)
        if len(all_frames) == 0:
            all_frames = frames
            all_labels = labels
        else:
            all_frames = np.vstack((all_frames, frames))
            all_labels = np.vstack((all_labels, labels))

    print('size: ', all_frames.shape)
    return all_frames, all_labels

def main3():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', '-g', type=int, default=0)
    parser.add_argument('--batch_size', '-b', type=int, default=60)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--real_test',  dest='real_test', action='store_true', 
                                help='Whether to split the data or use a complete new trial.')
    parser.add_argument('--max_epoch', '-e', type=int, default=110)
    parser.add_argument('--resume', '-r', type=int, default=None)
    parser.add_argument('--out_dir', '-o', type=str, default='/mnt/7ac4c5b9-8c05-451f-9e6d-897daecb7442/gears/results_gsm/result_right_arm2')
    args = parser.parse_args()

    model = GoalScoreModel()

    frames, labels = load_all_data(prep_f=model.prepare)

    frames, labels = igp.unison_shuffled_copies(frames, labels)
    print('Frames shape: ', frames.shape, ' Labels shape: ', labels.shape)

    data = chainer.datasets.TupleDataset(frames, labels)#.to_device(gpu_id)
    print('Dataset length: ', data._length)

    print('Frame size: ', data[0][0].shape, data[0][0].dtype)

    if args.real_test:
        print('Using test trial.')
        train_iter = iterators.SerialIterator(data, args.batch_size, shuffle=True)

        # Load the test data
        test_frames, test_labels = load_frames_labels(ids=[11], filestype=''.join((args.data_base_dir, args.data_file_pattern)), blackout=args.blackout)
        data_test = chainer.datasets.TupleDataset(test_frames, test_labels)
        test_iter = iterators.SerialIterator(data_test, args.batch_size, repeat=False, shuffle=False)
    else:   
        data_test, data_train = split_dataset(data, int(args.test_split*len(data)))
        train_iter = iterators.SerialIterator(data_train, args.batch_size, shuffle=True)
        test_iter = iterators.SerialIterator(data_test, args.batch_size, repeat=False, shuffle=False)


    if args.gpu_id >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu(args.gpu_id)

    # Create the optimizer for the model
    optimizer = optimizers.Adam().setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=1e-6))

    updater = training.StandardUpdater(train_iter, optimizer, 
                                       loss_func=model.calc_loss,
                                       device=args.gpu_id)


    # Full training
    print('Full model training ...')
    trainer = training.Trainer(updater, (args.max_epoch, 'epoch'), out=args.out_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, eval_func=model.calc_loss, device=args.gpu_id), name='val', trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/mae', 'main/gnll', 'main/weighted', 'main/VAE', 'main/VAE_REC','main/VAE_KL', 'val/main/loss', 'val/main/mae', 'val/main/weighted', 'elapsed_time']))#, 'val/main/VAE', 'main/loss', 'validation/main/loss', 'elapsed_time'], ))
    trainer.extend(extensions.PlotReport(['main/mae', 'val/main/mae', 'main/VAE', 'val/main/VAE'], x_key='epoch', file_name='loss.png', marker=None))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.FailOnNonNumber())
    # Save every X epochs
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.trainer'), trigger=(200, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, '%s_model_epoch_{.updater.epoch}.model' % (model.__class__.__name__)), trigger=(10, 'epoch'))
    
    trainer.extend(utils.display_image(model.vae_image, data_test, args.out_dir, args.gpu_id, n=3), trigger=(1, 'epoch'))

    trainer.extend(extensions.ExponentialShift('alpha', 0.5, init=1e-3, target=1e-8), trigger=(100, 'epoch'))

    # Resume from a specified snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
    print('Done.')


## For plots

import matplotlib.pyplot as plt
colours = ['steelblue', 'firebrick', 'darkseagreen']
lss = ['-', '--', '-.', ':']
import chainercv, cupy

def plot_data(filename, mus, vars, names):
    t = np.linspace(0, 1, num=84)
    # t = np.linspace(0, 1, num=len(mus[0]))

    fig, ax = plt.subplots(1)
    ax.plot(t, lw=1, color='black', ls='--', alpha=0.1) #label='target mean', 
    if len(mus[0]) > 84*13./40:
        plt.axvline(x=84*13./40, color='red', ls='--', alpha=0.5)
    if len(mus[0]) > 84*17./40:
        plt.axvline(x=84*17./40, color='red', ls='--', alpha=0.5)


    plt.text(x=1, y=0.45, s='Gear Pick Up', fontsize=12)
    if len(mus[0]) > 84*13./40:
        plt.text(x=35*0.8, y=0.1, s='Move\nGear', fontsize=12)
    if len(mus[0]) > 84*17./40:
        plt.text(x=50*0.8, y=0.45, s='Gear Insert', fontsize=12)

    for mu, var, name, c, ls in zip(mus, vars, names, colours, lss):
        sigma = np.sqrt(var)

        # the `n_sigma` sigma upper and lower analytic population bounds
        lower_bound3 = mu - 3 * sigma
        upper_bound3 = mu + 3 * sigma

        n_sigma = 1
        lower_bound_n = mu - n_sigma * sigma
        upper_bound_n = mu + n_sigma * sigma

        ax.plot(list(range(len(mus[0]))), mu, label='{} mean'.format(name), color=c, ls=ls)
        ax.fill_between(list(range(len(mus[0]))), lower_bound_n, upper_bound_n, facecolor=c, alpha=0.5
                    , label='{} ({} sigma range)'.format(name, n_sigma))
        ax.fill_between(list(range(len(mus[0]))), lower_bound3, upper_bound3, facecolor=c, alpha=0.2
                    , label='{} (3 sigma range)'.format(name))

    ax.legend(loc='upper left')
    ax.set_title('Goal Score Model performance')
    ax.set_ylabel('goal score estimate')
    ax.set_xlabel('timesteps')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 84)
    # ax.set_xlim(0, len(t))
    ax.grid()
    plt.tight_layout()
    plt.savefig(filename,  dpi=1600)
    plt.close()


def plot_graphs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', '-g', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--out_dir', '-o', type=str, default='/mnt/7ac4c5b9-8c05-451f-9e6d-897daecb7442/gears/results_gsm/result_right_arm2')
    args = parser.parse_args()

    model = GoalScoreModel()
    model.load_model(os.path.join(args.out_dir, 'GoalScoreModel_model_epoch_100.model'))
    model.to_gpu(args.gpu_id)
    chainer.backends.cuda.get_device_from_id(args.gpu_id).use()

    # Load data folder_id
    frames, labels = load_frames(folder_id=6, prep_f=model.prepare)
    
    frames_ds = chainer.dataset.to_device(args.gpu_id, frames[::10])

    a_mu, a_ln_var = model.forward_with_output_z(frames_ds[:args.batch_size])
    a_mu.to_cpu()
    a_ln_var.to_cpu()

    plot_data(os.path.join(args.out_dir, 'test.svg'), [a_mu.array.flatten()], [F.exp(a_ln_var).array.flatten()], ['right forearm'])


def plot_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', '-g', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--out_dir', '-o', type=str, default='/mnt/7ac4c5b9-8c05-451f-9e6d-897daecb7442/gears/results_gsm/result_test')
    args = parser.parse_args()

    model = GoalScoreModel()
    model.load_model(os.path.join(args.out_dir, 'GoalScoreModel_model_epoch_100.model'))
    model.to_gpu(args.gpu_id)
    chainer.backends.cuda.get_device_from_id(args.gpu_id).use()

    # Load data folder_id
    frames, labels = load_frames(folder_id=6, prep_f=model.prepare)
    
    frames_ds = chainer.dataset.to_device(args.gpu_id, frames[::10])

    a_mu, a_ln_var = model.forward_with_output_z(frames_ds[:args.batch_size])
    a_mu.to_cpu()
    a_ln_var.to_cpu()

    root_dir = os.path.join(args.out_dir, 'video')
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for i in range(len(a_mu)): 
        plot_data(os.path.join(root_dir, 'frame_{}.svg'.format(str(i).zfill(4))),
             [a_mu.array.flatten()[:i]], [F.exp(a_ln_var).array.flatten()[:i]], ['head camera'])


def show_vid():
    frames, labels = load_frames(folder_id=6, prep_f=None)

    for i, f in enumerate(frames):
        cv2.imshow('frame_{}'.format(i), f)
        cv2.waitKey(2)

if __name__ == '__main__':
    # main3()
    # plot_graphs()
    plot_video()
    # show_vid()

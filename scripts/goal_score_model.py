import numpy as np
import argparse

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

import math

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from mdn import MDN

class GoalScoreModel(chainer.Chain):
    """GoalScoreModel - ResNet frontend with an MDN backend. """
    def __init__(self, arg=None):
        super(GoalScoreModel, self).__init__()
        self.arg = arg
        with self.init_scope():

            self.head_model = ResNet50Layers()
            self.head_model.pick = 'pool5'
            self.head_model.layer_size = 2048
            self.head_model.feature = \
                    lambda x: self.head_model.forward(x, 
                              layers=[self.head_model.pick])[self.head_model.pick]

            self.mdn_hidden_units = 100
            self.mdn_gaussian_mixtures = 10
            self.mdn_input_dim = 1
            self.mdn_model = MDN(self.mdn_input_dim, self.mdn_hidden_units, self.mdn_gaussian_mixtures)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # print(x.shape)

        h = self.head_model.feature(x)

        # h2 = self.mdn_model.sample(h)
        h2 = self.mdn_model.sample_distribution(h)

        return h2


    def calc_loss(self, x, t):
        h = self.head_model.feature(x)
        # print('head_space', h)

        mu, log_var = self.forward(x)
        self.neg_log_like_loss = F.gaussian_nll(t, mu, log_var) # returns the sum of nll's

        z = F.gaussian(mu, log_var)
        self.mean_abs_error = F.mean_absolute_error(t, z)

        chainer.report({'nll': self.neg_log_like_loss}, self)
        chainer.report({'mae': self.mean_abs_error}, self)
        chainer.report({'sigma': F.mean(F.sqrt(F.exp(log_var)))}, self)

        self.total_loss = 0.1*self.mean_abs_error + \
                          (self.neg_log_like_loss / len(x))

        chainer.report({'loss': self.total_loss}, self)
        return self.total_loss


    def load_model(self, filename='my.model'):
        serializers.load_npz(filename, self)
        print('Loaded `{}` model.'.format(filename))

####

import cv2
def load_data(filename, size=(128, 128), data_size=100, verbose=0, blackout=True):
    cap = cv2.VideoCapture(filename)

    if verbose > 0:
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 800,800)

    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if blackout:
                frame = cv2.rectangle(frame,(0, 0),(200, 480), (0,0,0), -1)

            if verbose > 0:
                print(frame.shape)
                cv2.imshow('frame',frame)
            # frame = cv2.resize(frame, size)
            # frame = frame.astype(np.float32)/255.
            # frame = cv2.resize(frame, size).transpose((2, 1, 0))
            frame = chainer.links.model.vision.resnet.prepare(frame, size)
            frames.append(frame)
        else:
            break
        if cv2.waitKey(0 ) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    if verbose > 0:
        cv2.destroyAllWindows()
    if len(frames) > 0:
        print('Done loading video ({}).'.format(len(frames)))
    else:
        print('Didn\'t load any frames. Please check file ', filename, ' exists.')

    idx = list(map(int, np.linspace(0, len(frames), data_size, endpoint=False)))
    return np.take(frames, idx, axis=0).astype(np.float32)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_frames_labels(ids=list(range(10)), data_size=500, filestype='trial{}.avi', verbose=0):
    all_frames = []
    all_labels = []
    # print('Ids: ', ids)
    for i in ids:
        frames = load_data(filename=filestype.format(i), data_size=data_size, verbose=verbose)
        labels = np.linspace(0, 1, len(frames)).astype(np.float32).reshape(-1, 1)
        print(frames.shape, len(all_frames))
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
    parser.add_argument('--gpu_id', '-g', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=100)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--real_test',  dest='real_test', action='store_true', 
                                help='Whether to split the data or use a complete new trial.')
    parser.add_argument('--mdn_hidden-units', '-u', type=int, default=24)
    parser.add_argument('--mdn_gaussian-mixtures', '-m', type=int, default=24)
    parser.add_argument('--max_epoch', '-e', type=int, default=250)
    parser.add_argument('--resume', '-r', type=int, default=None)
    parser.add_argument('--out_dir', '-o', type=str, default='results/result_test')
    parser.add_argument('--data_base_dir', type=str, default='/media/daniel/data/hhc/')
    parser.add_argument('--data_file_pattern', '-f', type=str, default='trial{}.avi')
    args = parser.parse_args()


    # frames, labels = load_frames_labels(filestype='/media/daniel/data/hhc/trial{}_r_forearm.avi')
    frames, labels = load_frames_labels(filestype=''.join((args.data_base_dir, args.data_file_pattern)), verbose=0)

    frames, labels = unison_shuffled_copies(frames, labels)
    print('Frames shape: ', frames.shape, ' Labels shape: ', labels.shape)

    data = chainer.datasets.TupleDataset(frames, labels)#.to_device(gpu_id)
    print('Dataset length: ', data._length)

    print('Frame size: ', data[0][0].shape, data[0][0].dtype)

    if args.real_test:
        print('Using test trial.')
        train_iter = iterators.SerialIterator(data, args.batch_size, shuffle=True)

        # Load the test data
        test_frames, test_labels = load_frames_labels(ids=[11], filestype=''.join((args.data_base_dir, args.data_file_pattern)))
        test_data = chainer.datasets.TupleDataset(test_frames, test_labels)
        test_iter = iterators.SerialIterator(test_data, args.batch_size, repeat=False, shuffle=False)
    else:   
        data_test, data_train = split_dataset(data, int(args.test_split*len(data)))
        train_iter = iterators.SerialIterator(data_train, args.batch_size, shuffle=True)
        test_iter = iterators.SerialIterator(data_test, args.batch_size, repeat=False, shuffle=False)

    model = GoalScoreModel()

    if args.gpu_id >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu(args.gpu_id)
        # labels = chainer.dataset.to_device(args.gpu_id, labels)
        # frames = chainer.dataset.to_device(args.gpu_id, frames)


    # Create the optimizer for the model
    optimizer = optimizers.Adam().setup(model)
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

    # updater = training.ParallelUpdater(train_iter, optimizer,
    #                                 loss_func=model.calc_loss,
    #                                 devices={'main': args.gpu_id, 'second': 1})

    # Pre-training
    print('Pretraining started.')
    trainer = training.Trainer(updater, (3, 'epoch'), out=args.out_dir)
    # Disable update for the head model
    print('Disabling training of head model.')
    model.head_model.disable_update()
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.FailOnNonNumber())
    trainer.run()

    # Full training
    print('Full model training ...')
    trainer = training.Trainer(updater, (args.max_epoch, 'epoch'), out=args.out_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, eval_func=model.calc_loss, device=args.gpu_id), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/nll', 'main/mae', 'main/sigma' ,'validation/main/loss', 'validation/main/mae', 'validation/main/sigma', 'elapsed_time']))#, 'main/loss', 'validation/main/loss', 'elapsed_time'], ))
    trainer.extend(extensions.PlotReport(['main/mae', 'validation/main/mae'], x_key='epoch', file_name='loss.png', marker=None))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.FailOnNonNumber())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'), trigger=(20, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'model_epoch_{.updater.epoch}.model'), trigger=(20, 'epoch'))

    # Disable/Enable update for the head model
    model.head_model.enable_update()

    # Resume from a specified snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
    print('Done.')


def main2():
    model = GoalScoreModel()
    imgs = np.random.randn(1, 3, 128*2, 128*2).astype(np.float32)
    output = model(imgs)
    # print(output.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--input-dim', '-d', type=int, default=1)
    parser.add_argument('--n-samples', '-n', type=int, default=1000)
    parser.add_argument('--hidden-units', '-u', type=int, default=24)
    parser.add_argument('--gaussian-mixtures', '-m', type=int, default=24)
    parser.add_argument('--epoch', '-e', type=int, default=10000)
    args = parser.parse_args()

    model = ResNet50(pretrained_model='imagenet', arch='he')
    # By default, __call__ returns a probability score (after Softmax).
    imgs = np.random.randn(1, 3, 128, 128).astype(np.float32)
    prob = model(imgs)
    model.pick = 'res5'
    # This is layer res5
    res5 = model(imgs)

    print(model.__dict__)
    print('----')
    print(model['res5'].b2.__dict__)

    print('----')
    print(model['pool5'].__dict__)
    # model.pick = ['res5', 'fc6', 'pool5']
    # print(res5.shape) # (1, 2048, 4, 4)
    # print(fc6.shape)  # (1, 1000)
    # print(pool5.shape)    # (1, 2048)
    # # These are layers res5 and fc6.
    # res5, fc6, pool5 = model(imgs)


    # print(prob)
    #print(fc7.shape)
    print(model.layer_names)



if __name__ == '__main__':
    main3()

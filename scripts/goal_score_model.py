import numpy as np
import argparse

from models.mdn.mdn import MDN

import chainer
import chainer.functions as F
import chainer.links as L


from chainer.training.extensions import LogReport
from chainer import iterators
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extensions
from chainer.datasets import split_dataset
from chainer import optimizers

from chainer.links import ResNet50Layers

class GoalScoreModel(chainer.Chain):
    """docstring for GoalScoreModel"""
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

            self.mdn_hidden_units = 500
            self.mdn_gaussian_mixtures = 1
            self.mdn_input_dim = 1
            self.mdn_model = MDN(self.mdn_input_dim, self.mdn_hidden_units, self.mdn_gaussian_mixtures)
            self.mdn_model.sample_distribution = self._distrib
 

    def _distrib(self, x):
        pi, mu, log_var = self.mdn_model.get_gaussian_params(x)
        n_batch = pi.shape[0]

        # Choose one of Gaussian means and vars n_batch times
        ps = chainer.backends.cuda.to_cpu(pi.array)
        if np.any(np.isnan(ps)):
            print('Found nan values, aborting.', ps, ps.shape)
            exit(0)
        
        idx = [np.random.choice(self.mdn_model.gaussian_mixtures, p=p) for p in ps]
        # print(idx)
        # print(mu, mu.shape)
        # print(F.get_item(mu, (list(range(n_batch)), idx)))

        mu = F.get_item(mu, [list(range(n_batch)), idx])
        log_var = F.get_item(log_var, [list(range(n_batch)), idx])
        return mu, log_var

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # print(x.shape)

        h = self.head_model.feature(x)
        # h2 = self.mdn_model.sample(h)
        h2 = self.mdn_model.sample_distribution(h)
        # print(h2[0].shape)
        return h2


    def calc_loss(self, x, t):
        h = self.head_model.feature(x)
        # print('head_space', h)
        self.neg_log_like_loss = self.mdn_model.negative_log_likelihood(h, t)

        # print('neg_log_like_Loss: ', self.neg_log_like_loss)
        mu, log_var = self.forward(x)
        z = F.gaussian(mu, log_var)
        self.mean_abs_error = F.mean_absolute_error(t, z)

        chainer.report({'loss': self.neg_log_like_loss, 'mean_abs_error': self.mean_abs_error}, self)
        
        # return self.loss
        return self.mean_abs_error + 0.1*self.neg_log_like_loss

####

def load_data(filename, size=(128, 128), verbose=0):
    import cv2
    cap = cv2.VideoCapture(filename)
    if verbose > 0:
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 800,800)

    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
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
    print('Done loading video.')
    return np.asarray(frames).astype(np.float32)


def main3():
    batch_size = 1
    test_split = 0.1
    gpu_id = 0
    max_epoch = 10000
    
    frames = load_data(filename='/media/daniel/data/hhc/VID_20181210_150814.mp4', verbose=0)
    labels = np.linspace(0, 1, len(frames)).astype(np.float32).reshape(-1, 1)
    print(frames.shape, labels.shape)

    data = chainer.datasets.TupleDataset(frames, labels)#.to_device(gpu_id)
    print(data._length)

    print('Frame size: ', data[0][0].shape, data[0][0].dtype)
    
    data_test, data_train = split_dataset(data, int(test_split*len(data)))
    train_iter = iterators.SerialIterator(data_train, batch_size, shuffle=False)
    test_iter = iterators.SerialIterator(data_test, batch_size, repeat=False, shuffle=False)

     
    model = GoalScoreModel()



    if gpu_id >= 0:
        chainer.backends.cuda.get_device_from_id(gpu_id).use()
        model.to_gpu(gpu_id)
        chainer.dataset.to_device(gpu_id, labels)
        chainer.dataset.to_device(gpu_id, frames)


    # Create the optimizer for the model
    optimizer = optimizers.Adam().setup(model)

    # xp = chainer.backend.get_array_module(data_train)
    # optimizer.update(model.calc_loss, xp.asarray([data_train[0][0]]), xp.asarray([data_train[0][1]]))
    # import chainer.computational_graph as c
    # g = c.build_computational_graph(model.neg_log_like_loss)
    # with open('result/graph.dot', 'w') as o:
    #     o.write(g.dump())

    updater = training.StandardUpdater(train_iter, optimizer, 
                                       loss_func=model.calc_loss,
                                       device=gpu_id)

    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model, eval_func=model.calc_loss, device=gpu_id), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/mean_abs_error' ,'validation/main/loss', 'validation/main/mean_abs_error', 'elapsed_time']))#, 'main/loss', 'validation/main/loss', 'elapsed_time'], ))
    # trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    # trainer.extend(extensions.dump_graph('main/loss'))
    # trainer.extend(extensions.ProgressBar())
    # trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'), trigger=(100, 'epoch'))

    model.head_model.disable_update()
    
    trainer.run()
    print('done.')


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

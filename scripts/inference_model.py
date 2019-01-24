import numpy as np
import argparse, os
import cv2

import goal_score_model as gsm
import chainer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colours = ['steelblue', 'firebrick', 'darkseagreen']
lss = ['-', '--', '-.', ':']

test_ids = [11]
gpu_id = 0

test_image = 30

def plot_image(filename, id, dataset, mu=None, var=None):
    img = chainer.dataset.to_device(-1, dataset[id])
    img = img.transpose((1, 2, 0))
    img += np.array([103.939, 116.779, 123.68], dtype=np.float32)
    print(img.shape)
    print(filename)
    cv2.imwrite(filename, img)


def plot_data(filename, mus, vars, names):
    t = np.linspace(0, 1, num=len(mus[0]))

    fig, ax = plt.subplots(1)
    ax.plot(t, lw=1, label='target mean', color='black', ls='--')

    for mu, var, name, c, ls in zip(mus, vars, names, colours, lss):
        sigma = np.sqrt(var)

        # the `n_sigma` sigma upper and lower analytic population bounds
        lower_bound3 = mu - 3 * sigma
        upper_bound3 = mu + 3 * sigma

        n_sigma = 1
        lower_bound_n = mu - n_sigma * sigma
        upper_bound_n = mu + n_sigma * sigma

        ax.plot(mu, label='{}_mean'.format(name), color=c, ls=ls)
        ax.fill_between(np.arange(len(mu)), lower_bound_n, upper_bound_n, facecolor=c, alpha=0.5)
                    # , label='{} {} sigma range'.format(name, n_sigma))
        ax.fill_between(np.arange(len(mu)), lower_bound3, upper_bound3, facecolor=c, alpha=0.2)
                    # , label='{} 3 sigma range'.format(name))

    ax.legend(loc='upper left')
    ax.set_title('Goal Score model')
    ax.set_ylabel('t_predicted')
    ax.set_xlabel('t_groundtruth')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(t))
    ax.grid()

    plt.savefig(filename)
    plt.close()

def find_last_model(path):
    print('To be done')
    pass


def test_kinect():
    base_path = 'results/result_kinect2'
    kinect_model = gsm.GoalScoreModel()
    kinect_model.load_model(os.path.join(base_path, 'model_epoch_200.model'))
    kinect_model.to_gpu(gpu_id)

    kinect_frames, kinect_labels = gsm.load_frames_labels(ids=test_ids, data_size=100, filestype='/media/daniel/data/hhc/trial{}_kinect2_qhd.avi')
    kinect_frames = chainer.dataset.to_device(gpu_id, kinect_frames)
    # # kinect_frames, kinect_labels = gsm.unison_shuffled_copies(kinect_frames, kinect_labels)
    print('frames shape: ', kinect_frames.shape, kinect_labels.shape)

    infered_mu, infered_logvar = kinect_model.forward(kinect_frames)
    
    infered_mu.to_cpu()
    infered_logvar.to_cpu()
    
    print('infered shape:', infered_mu.shape)
    # print(infered_mu.array)

    os.makedirs(os.path.join(base_path, 'inferences'), exist_ok=True)
    plot_data(os.path.join(base_path, 'inferences', 'kinect_{}.png'.format(test_ids)), [infered_mu.array.flatten()], [np.exp(infered_logvar.array.flatten())], ['kinect'])
    plot_image(os.path.join(base_path, 'inferences', 'kinect_{}_{}.png'.format(test_ids, test_image)), test_image, kinect_frames)

    return infered_mu.array.flatten(), infered_logvar.array.flatten()

def test_r_forearm():
    base_path = 'results/result_r_forearm'
    r_forearm_model = gsm.GoalScoreModel()
    r_forearm_model.load_model(os.path.join(base_path, 'model_epoch_200.model'))
    r_forearm_model.to_gpu(gpu_id)

    r_forearm_frames, r_forearm_labels = gsm.load_frames_labels(ids=test_ids, data_size=100, filestype='/media/daniel/data/hhc/trial{}_r_forearm.avi')
    print(r_forearm_frames.shape, r_forearm_labels.shape)
    r_forearm_frames = chainer.dataset.to_device(gpu_id, r_forearm_frames)
    # r_forearm_frames, r_forearm_labels = gsm.unison_shuffled_copies(r_forearm_frames, r_forearm_labels)

    infered_mu, infered_logvar = r_forearm_model.forward(r_forearm_frames)
    infered_mu.to_cpu()
    infered_logvar.to_cpu()

    print('infered shape:', infered_mu.shape)
    # print(infered_mu.array)

    os.makedirs(os.path.join(base_path, 'inferences'), exist_ok=True)
    plot_data(os.path.join(base_path, 'inferences', 'r_forearm_{}.png'.format(test_ids)), [infered_mu.array.flatten()], [np.exp(infered_logvar.array.flatten())], ['r_forearm'])
    plot_image(os.path.join(base_path, 'inferences', 'r_forearm_{}_{}.png'.format(test_ids, test_image)), test_image, r_forearm_frames)

    return infered_mu.array.flatten(), infered_logvar.array.flatten()

def test_l_forearm():
    base_path = 'results/result_l_forearm'
    l_forearm_model = gsm.GoalScoreModel()
    l_forearm_model.load_model(os.path.join(base_path, 'model_epoch_200.model'))
    l_forearm_model.to_gpu(gpu_id)

    l_forearm_frames, l_forearm_labels = gsm.load_frames_labels(ids=test_ids, data_size=100, filestype='/media/daniel/data/hhc/trial{}_l_forearm.avi')
    print(l_forearm_frames.shape, l_forearm_labels.shape)
    l_forearm_frames = chainer.dataset.to_device(gpu_id, l_forearm_frames)
    # r_forearm_frames, r_forearm_labels = gsm.unison_shuffled_copies(r_forearm_frames, r_forearm_labels)

    infered_mu, infered_logvar = l_forearm_model.forward(l_forearm_frames)
    infered_mu.to_cpu()
    infered_logvar.to_cpu()

    print('infered shape:', infered_mu.shape)
    # print(infered_mu.array)

    os.makedirs(os.path.join(base_path, 'inferences'), exist_ok=True)
    plot_data(os.path.join(base_path, 'inferences', 'l_forearm_{}.png'.format(test_ids)), [infered_mu.array.flatten()], [np.exp(infered_logvar.array.flatten())], ['l_forearm'])
    plot_image(os.path.join(base_path, 'inferences', 'l_forearm_{}_{}.png'.format(test_ids, test_image)), test_image, l_forearm_frames)

    return infered_mu.array.flatten(), infered_logvar.array.flatten()


def main(args):
    chainer.backends.cuda.get_device_from_id(gpu_id).use()

    for i in [11] + list(range(10)):
        global test_ids
        test_ids=[i]
        mu0, logvar0 = test_r_forearm()
        mu1, logvar1 = test_l_forearm()
        mu2, logvar2 = test_kinect()
        mus = [mu0, mu1, mu2]
        vars = [np.exp(logvar0), np.exp(logvar1), np.exp(logvar2)]
        plot_data('results/both_{}.png'.format(test_ids), mus, vars, ['r_forearm', 'l_forearm', 'kinect'])

if __name__ == '__main__':
    main({'model': 'all'})
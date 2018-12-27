import numpy as np
import argparse, os

import goal_score_model as gsm
import chainer

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colours = ['steelblue', 'firebrick', 'darkseagreen']
lss = ['-', '--', '-.', ':']

test_ids = [11]
gpu_id = 1

def plot_data(filename, mus, vars, names):
    t = np.arange(len(mus[0]))

    fig, ax = plt.subplots(1)
    ax.plot(t, t, lw=1, label='target mean', color='black', ls='--')

    for mu, var, name, c, ls in zip(mus, vars, names, colours, lss):
        sigma = np.sqrt(var)

        # the `n_sigma` sigma upper and lower analytic population bounds
        n_sigma = 3
        lower_bound = mu*t - n_sigma*sigma*np.sqrt(t)
        upper_bound = mu*t + n_sigma*sigma*np.sqrt(t)

        # print(colors.keys())
        # c = np.random.choice(list(colors.keys()))
        ax.plot(t, mu*t, label='{}_mean'.format(name), color=c, ls=ls)

        ax.fill_between(t, lower_bound, upper_bound, facecolor=c, alpha=0.5,
                    label='{} {} sigma range'.format(name, n_sigma))

    ax.legend(loc='upper left')
    ax.set_title('Goal Score model')
    ax.set_ylabel('t_predicted')
    ax.set_xlabel('t_groundtruth')
    ax.grid()

    plt.savefig(filename)
    plt.close()


def test_kinect():
    base_path = 'results/result_kinect_mse'
    kinect_model = gsm.GoalScoreModel()
    kinect_model.load_model(os.path.join(base_path, 'model_epoch_40_back.model'))
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

    return infered_mu.array.flatten(), infered_logvar.array.flatten()

def test_r_forearm():
    base_path = 'results/result_forearm'
    r_forearm_model = gsm.GoalScoreModel()
    r_forearm_model.load_model(os.path.join(base_path, 'model_epoch_r_forearm200.model'))
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
    plot_data(os.path.join(base_path, 'inferences/forearm_{}.png'.format(test_ids)), [infered_mu.array.flatten()], [np.exp(infered_logvar.array.flatten())], ['r_forearm'])

    return infered_mu.array.flatten(), infered_logvar.array.flatten()

def main(args):
    chainer.backends.cuda.get_device_from_id(gpu_id).use()

    for i in [11] + list(range(10)):
        global test_ids
        test_ids=[i]
        mu1, logvar1 = test_r_forearm()
        mu2, logvar2 = test_kinect()
        mus = [mu1, mu2]
        vars = [np.exp(logvar1), np.exp(logvar2)]
        plot_data('results/both_{}.png'.format(test_ids), mus, vars, ['r_forearm', 'kinect'])

if __name__ == '__main__':
    main({'model': 'all'})
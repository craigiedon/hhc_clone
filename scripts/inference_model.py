import numpy as np
import argparse

import goal_score_model as gsm


import matplotlib.pyplot as plt
def plot_data(name, mu, var):
    t = np.arange(len(mu))
    sigma = np.sqrt(var)

    # the 1 sigma upper and lower analytic population bounds
    n_sigma = 3
    lower_bound = mu*t - n_sigma*sigma*np.sqrt(t)
    upper_bound = mu*t + n_sigma*sigma*np.sqrt(t)

    print(lower_bound.shape)

    fig, ax = plt.subplots(1)

    ax.plot(t, mu*t, label='predicted_mean', color='blue', ls='-.')
    ax.plot(t, t, lw=1, label='target mean', color='black', ls='--')
    ax.fill_between(t, lower_bound, upper_bound, facecolor='blue', alpha=0.5,
                label='{} sigma range'.format(n_sigma))
    ax.legend(loc='upper left')
    ax.set_title('Goal Score model')
    ax.set_ylabel('t_predicted')
    ax.set_xlabel('t_groundtruth')
    ax.grid()

    plt.savefig(name)

import chainer

test_ids = [11]
gpu_id = 1

chainer.backends.cuda.get_device_from_id(gpu_id).use()

def test_kinect():
    kinect_model = gsm.GoalScoreModel()
    kinect_model.load_model('results_kinect/model_epoch_kinect.model')
    kinect_model.to_gpu(gpu_id)

    kinect_frames, kinect_labels = gsm.load_frames_labels(ids=test_ids, data_size=100, filestype='/media/daniel/data/hhc/trial{}_kinect2_qhd.avi')
    kinect_frames = chainer.dataset.to_device(gpu_id, kinect_frames)
    # # kinect_frames, kinect_labels = gsm.unison_shuffled_copies(kinect_frames, kinect_labels)
    print(kinect_frames.shape, kinect_labels.shape)

    infered_mu, infered_logvar = kinect_model.forward(kinect_frames)
    
    infered_mu.to_cpu()
    infered_logvar.to_cpu()
    
    print(infered_mu.shape)
    print(infered_mu.array)

    plot_data('results_kinect/inferences/kinect_{}.png'.format(test_ids), infered_mu.array.flatten(), np.exp(infered_logvar.array.flatten()))

def test_r_forearm():
    r_forearm_model = gsm.GoalScoreModel()
    r_forearm_model.load_model('result_forearm/model_epoch_r_forearm200.model')
    r_forearm_model.to_gpu(gpu_id)

    r_forearm_frames, r_forearm_labels = gsm.load_frames_labels(ids=test_ids, data_size=100, filestype='/media/daniel/data/hhc/trial{}_r_forearm.avi')
    print(r_forearm_frames.shape, r_forearm_labels.shape)
    r_forearm_frames = chainer.dataset.to_device(gpu_id, r_forearm_frames)
    # r_forearm_frames, r_forearm_labels = gsm.unison_shuffled_copies(r_forearm_frames, r_forearm_labels)

    infered_mu, infered_logvar = r_forearm_model.forward(r_forearm_frames)
    infered_mu.to_cpu()
    infered_logvar.to_cpu()
    print(infered_mu.shape)
    print(infered_mu.array)

    plot_data('result_forearm/inferences/forearm_{}.png'.format(test_ids), infered_mu.array.flatten(), np.exp(infered_logvar.array.flatten()))


def main(args):
    for i in range(10):
        global test_ids
        test_ids=[i]
        # test_r_forearm()
        test_kinect()

    # kinect_model = gsm.GoalScoreModel()
    # kinect_model.load_model('result_kinect/model_epoch_kinect.model')
    # kinect_model.to_gpu(gpu_id)

    # # r_forearm_model = gsm.GoalScoreModel()
    # # r_forearm_model.load_model('result_forearm/model_epoch_r_forearm200.model')
    # # r_forearm_model.to_gpu(gpu_id)

    # # kinect_frames, kinect_labels = gsm.load_frames_labels(ids=test_ids, filestype='/media/daniel/data/hhc/trial{}_kinect2_qhd.avi')
    # # # kinect_frames, kinect_labels = gsm.unison_shuffled_copies(kinect_frames, kinect_labels)
    # # print(kinect_frames.shape, kinect_labels.shape)

    # r_forearm_frames, r_forearm_labels = gsm.load_frames_labels(ids=test_ids, data_size=100, filestype='/media/daniel/data/hhc/trial{}_r_forearm.avi')
    # print(r_forearm_frames.shape, r_forearm_labels.shape)
    # r_forearm_frames = chainer.dataset.to_device(gpu_id, r_forearm_frames)
    # # chainer.dataset.to_device(gpu_id, labels)
    # # r_forearm_frames, r_forearm_labels = gsm.unison_shuffled_copies(r_forearm_frames, r_forearm_labels)

    # # infered = kinect_model.forward(kinect_frames)
    # infered_mu, infered_logvar = r_forearm_model.forward(r_forearm_frames)
    # infered_mu.to_cpu()
    # infered_logvar.to_cpu()
    # print(infered_mu.shape)
    # print(infered_mu.array)

    # plot_data('result_forearm/inferences/forearm_{}.png'.format(test_ids), infered_mu.array.flatten(), np.exp(infered_logvar.array.flatten()))

if __name__ == '__main__':
    main({'model': 'kinect2'})
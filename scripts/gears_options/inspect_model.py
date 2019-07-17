
import cv2
import glob
from insert_gear_policy import *
import numpy as np
import argparse
import chainer, cupy, chainercv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', '-g', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=40)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--real_test',  dest='real_test', action='store_true', 
                                help='Whether to split the data or use a complete new trial.')
    parser.add_argument('--max_epoch', '-e', type=int, default=2500)
    parser.add_argument('--resume', '-r', type=int, default=None)
    parser.add_argument('--out_dir', '-o', type=str, default='results/result_test')
    args = parser.parse_args()


    model = InsertGearPolicyResNet()
    # Load the model
    # model.load_model(filename='/mnt/7ac4c5b9-8c05-451f-9e6d-897daecb7442/gears/results/results_scaled/model_epoch_300.model')
    model.load_model(filename='/mnt/7ac4c5b9-8c05-451f-9e6d-897daecb7442/gears/results/results_scaled_mdn/model_epoch_20.model')

    # frames, joints = load_data(prep_f=model.prepare, prepare_joints=model.prepare_joints)
    frames, joints = load_data(prep_f=model.prepare, prepare_joints=None, subset=2)


    gpu_id=0
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
        frames = chainer.dataset.to_device(gpu_id, frames)

    data_iter = iterators.SerialIterator(frames, args.batch_size, shuffle=False)

    # data = chainer.datasets.TupleDataset(frames, joints)
    # print('Dataset length: ', data._length)


    # WORKING BELOW
    output_joints = []
    error_joints = []
    # print('Output:', output_joints.shape, output_joints)
    np.set_printoptions(precision=4, linewidth=np.inf)
    for f, t in zip(frames, joints):
        out = model.forward(cupy.asarray([f]))[0]
        y = chainer.cuda.to_cpu(out.data)
        print(' {} {} {} {}'.format(y, t, abs(y-t), abs(y-t)* model.JOINT_SCALES_RANGE))
        output_joints.append(list(y))
        error_joints.append(list(abs(y-t)))

    output_joints=np.asarray(output_joints)
    error_joints=np.asarray(error_joints)


    # # TESTING THIS
    # in_values, out_values, rest_values = chainercv.utils.apply_to_iterator(
    #         model.forward, data_iter)

    # print(out_values.shape)

    # exit(0)
    # try:
    #     for batch in data_iter:
    #         print(len(batch))
    #         # print(type(batch), batch[0][0].shape, batch[0][1].shape)
    #         out = model.forward(cupy.asarray(batch))
    #         out = chainer.cuda.to_cpu(out.data)
    #         # print(out, output_joints)
    #         output_joints = np.concatenate((output_joints, out), axis=0)
    #         print(out.shape, output_joints.shape, type(out), type(output_joints))
    # except StopIteration, AttributeError:
    #     print('stopped?')

    print(output_joints.shape)

    # for y, t in zip(output_joints, joints):
    #     print(' {} {} {}'.format(y, t, abs(y-t)))

    # frames, joints = unison_shuffled_copies(frames, joints)
    # print('joints shape: ', joints.shape)

    # print('joint min/max', np.min(joints[:,0]), np.max(joints[:,0]))
    # print('joint min/max', np.min(joints[:,1]), np.max(joints[:,1]))
    # print('joint min/max', np.min(joints[:,2]), np.max(joints[:,2]))
    # print('joint min/max', np.min(joints[:,3]), np.max(joints[:,3]))
    # print('joint min/max', np.min(joints[:,4]), np.max(joints[:,4]))
    # print('joint min/max', np.min(joints[:,5]), np.max(joints[:,5]))
    # print('joint min/max', np.min(joints[:,6]), np.max(joints[:,6]))

    error_joints *= model.JOINT_SCALES_RANGE
    import matplotlib.pyplot as plt
    plt.plot(sorted(error_joints[:,0]), 'r.')
    plt.plot(sorted(error_joints[:,1]), 'g.')
    plt.plot(sorted(error_joints[:,2]), 'b.')
    plt.plot(sorted(error_joints[:,3]), '.')
    plt.plot(sorted(error_joints[:,4]), '.')
    plt.plot(sorted(error_joints[:,5]), '.')
    plt.plot(sorted(error_joints[:,6]), '.')
    plt.title('Error between true and pred (in rads)')
    # plt.plot(sorted(joints[:,0]), 'r.')
    # plt.plot(sorted(joints[:,1]), 'g.')
    # plt.plot(sorted(joints[:,2]), 'b.')
    # plt.plot(sorted(joints[:,3]), '.')
    # plt.plot(sorted(joints[:,4]), '.')
    # plt.plot(sorted(joints[:,5]), '.')
    # plt.plot(sorted(joints[:,6]), '.')
    # plt.plot(sorted(output_joints[:,0]), 'ro')
    # plt.plot(sorted(output_joints[:,1]), 'go')
    # plt.plot(sorted(output_joints[:,2]), 'bo')
    # plt.plot(sorted(output_joints[:,3]), 'o')
    # plt.plot(sorted(output_joints[:,4]), 'o')
    # plt.plot(sorted(output_joints[:,5]), 'o')
    # plt.plot(sorted(output_joints[:,6]), 'o')
    # plt.title('Distribution of joint angles per joint')
    plt.xlabel('joints')
    plt.ylabel('rad')
    plt.grid(True)
    plt.xlim(0, len(joints))
    plt.legend(['joint'+str(i) for i in range(7)])
    plt.tight_layout()
    plt.show()

    # data = chainer.datasets.TupleDataset(frames, joints)
    # print('Dataset length: ', data._length)

    # print('Frame size: ', data[0][0].shape, data[0][0].dtype)

    # if args.real_test:
    #     print('Using test trial.')
    #     train_iter = iterators.SerialIterator(data, args.batch_size, shuffle=True)

    #     # Load the test data
    #     test_frames, test_joints = load_frames_labels(ids=[11], filestype=''.join((args.data_base_dir, args.data_file_pattern)), blackout=args.blackout)
    #     data_test = chainer.datasets.TupleDataset(test_frames, test_joints)
    #     test_iter = iterators.SerialIterator(data_test, args.batch_size, repeat=False, shuffle=False)
    # else:
    #     print('Splitting data at ratio: ', args.test_split)
    #     data_test, data_train = split_dataset(data, int(args.test_split*len(data)))
    #     train_iter = iterators.SerialIterator(data_train, args.batch_size, shuffle=True)
    #     test_iter = iterators.SerialIterator(data_test, args.batch_size, repeat=False, shuffle=False)




if __name__ == '__main__':
    main()
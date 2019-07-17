
import cv2
import glob
from insert_gear_policy import *
import numpy as np
import argparse
import chainercv

IMAGE_SIZE = 128


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
    joints = np.asarray(load_joints())


    joints = model.prepare_joints(joints)

    # frames, joints = unison_shuffled_copies(frames, joints)
    print('joints shape: ', joints.shape)

    print('joint min/max', np.min(joints[:,0]), np.max(joints[:,0]))
    print('joint min/max', np.min(joints[:,1]), np.max(joints[:,1]))
    print('joint min/max', np.min(joints[:,2]), np.max(joints[:,2]))
    print('joint min/max', np.min(joints[:,3]), np.max(joints[:,3]))
    print('joint min/max', np.min(joints[:,4]), np.max(joints[:,4]))
    print('joint min/max', np.min(joints[:,5]), np.max(joints[:,5]))
    print('joint min/max', np.min(joints[:,6]), np.max(joints[:,6]))

    import matplotlib.pyplot as plt
    plt.plot(sorted(joints[:,0]), 'r.')
    plt.plot(sorted(joints[:,1]), 'g.')
    plt.plot(sorted(joints[:,2]), 'b.')
    plt.plot(sorted(joints[:,3]), '.')
    plt.plot(sorted(joints[:,4]), '.')
    plt.plot(sorted(joints[:,5]), '.')
    plt.plot(sorted(joints[:,6]), '.')
    plt.title('Distribution of joint angles per joint')
    plt.xlabel('joints')
    plt.ylabel('rad')
    plt.grid(True)
    plt.xlim(0, len(joints))
    plt.legend(['joint'+str(i) for i in range(7)])
    plt.tight_layout()
    plt.show()


    # Inspace images
    frames = load_frames(skip_last=True, prep_f=model.prepare, subset=3)
    skipcount=5
    frames = frames[::skipcount]

    frames, joints = colour_agumentations(frames, joints[:len(frames)], n=2)

    for f in frames:
        chainercv.visualizations.vis_image(f*255)
        plt.show()



if __name__ == '__main__':
    main()
import glob
from tqdm import tqdm
import numpy as np 

def load_joints(filename=None, min_joint=31, max_joint=38, verbose=1, skip_first=True, subset=-1):
    folders = sorted(glob.glob('data/demo_*'))

    if subset >= 0:
        folders = folders[:subset]

    for folder in tqdm(folders):
        files = sorted(glob.glob(folder + '/joint_position_*'))
        skipped = False

        joints = []
        if verbose > 0:
            print(folder + '/joint_position_*', files)

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

        joints = np.asarray(joints)
        print('joints.shape ', joints.shape)
        import matplotlib.pyplot as plt
        plt.plot(joints[:,0], 'r.')
        plt.plot(joints[:,1], 'g.')
        plt.plot(joints[:,2], 'b.')
        plt.plot(joints[:,3], '.')
        plt.plot(joints[:,4], '.')
        plt.plot(joints[:,5], '.')
        plt.plot(joints[:,6], '.')
        plt.title('Distribution of joint angles per joint')
        plt.xlabel('joints')
        plt.ylabel('rad')
        plt.grid(True)
        plt.xlim(0, len(joints))
        plt.legend(['joint'+str(i) for i in range(7)])
        plt.tight_layout()
        plt.show()
    

    # When everything done, release the capture
    if len(joints) > 0:
        print('Done loading joints data. ({}).'.format(len(joints)))
    else:
        print('Didn\'t load any joints. Please check file ', filename, ' exists.')

    return np.asarray(joints)

def load_position(filename=None, min_joint=0, max_joint=3, verbose=0, skip_first=True, subset=-1):
    folders = sorted(glob.glob('data/demo_*'))

    if subset >= 0:
        folders = folders[:subset]

    for folder in tqdm(folders):
        files = sorted(glob.glob(folder + '/l_wrist_roll_link_*'))
        skipped = False

        joints = []
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

        joints = np.asarray(joints)
        print('joints.shape ', joints.shape)
        import matplotlib.pyplot as plt
        plt.plot(joints[:,0], 'r.')
        plt.plot(joints[:,1], 'g.')
        plt.plot(joints[:,2], 'b.')
        plt.title('Distribution of joint angles per joint')
        plt.xlabel('samples')
        plt.ylabel('displacement')
        plt.grid(True)
        plt.xlim(0, len(joints))
        plt.legend(['x', 'y', 'z'])
        plt.tight_layout()
        plt.show()
    

    # When everything done, release the capture
    if len(joints) > 0:
        print('Done loading joints data. ({}).'.format(len(joints)))
    else:
        print('Didn\'t load any joints. Please check file ', filename, ' exists.')

    return np.asarray(joints)



def main():
    # load_joints()
    load_position()

if __name__ == '__main__':
    main()

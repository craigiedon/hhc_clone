import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import robotcontroller as rc
import math
import numpy as np

class PickupGear(object):
    """docstring for Option1: Grab gear"""
    def __init__(self, pr2_left=None, pr2_right=None):
        super(PickupGear, self).__init__()
        if pr2_left is None:
            self.pr2_left = rc.PR2RobotController('left_arm', add_table=True)
        else:
            self.pr2_left = pr2_left

        # self.pr2.reset_pose()


        self.action_sequence_p = [
                                 ['remove_and_deattach_gears'],
                                 ['close_gripper'],
                                 [0.20, 0.7, 0.88],
                                 [0.50, 0.7, 0.92],
                                 [0.50, 0.7, 0.88], # down
                                 [0.42, 0.7, 0.88], # pull towards
                                 [0.42, 0.7, 0.92],
                                 [0.25, 0.7, 0.92],
                                 [0.25, 0.7, 0.73],
                                 ['open_gripper'],
                                 [0.31, 0.685, 0.73],
                                 ['close_gripper'],
                                 [0.20, 0.7, 0.73],
                                 ['add_and_attach_gears'],
                                 ]
        self.action_sequence_rpy = [
                                    ['gear'],
                                    [50],
                                    [0, 0 , 0],
                                    [0, math.pi/4., 0],
                                    [0, math.pi/4., 0],
                                    [0, math.pi/4., 0],
                                    [0, math.pi/4., 0],
                                    [0, math.pi/4., 0],
                                    [math.pi/2., 0, 0],
                                    [0.03],
                                    [math.pi/2., 0, 0],
                                    [50],
                                    [math.pi/2., 0, 0],
                                    ['gear'],
                                    ]

        assert(len(self.action_sequence_p) == len(self.action_sequence_rpy))
        self.current_action = 0
        self.dist_thr = 0.2


    def in_initiation_set(self, state=None):
        if state is None:
            current_pose = self.pr2_left.get_current_pose()
            print('Current pose', current_pose)
        else:
            current_pose = state

        seq_id = 0

        close_seq_ids = []

        while True:
            if (seq_id >= len(self.action_sequence_p)):
                break
            if (len(self.action_sequence_p[seq_id]) != 3):
                seq_id += 1
                continue

            # print(self.action_sequence_p[seq_id])

            v = [getattr(current_pose.position, k) - self.action_sequence_p[seq_id][i] for i, k in enumerate(['x', 'y', 'z'])]
            # find norm of v
            dv = np.linalg.norm(v)
            if dv > self.dist_thr:
                pass
                # print('Current location is too far from any of the target locations', dv, self.dist_thr)
                # return False
            else:
                close_seq_ids.append([seq_id, dv])
            
            seq_id += 1

        print('Close seq ids: ', close_seq_ids)
        if len(close_seq_ids) > 0:
            print('We found a close state, so we are in initiation set.')
            return True
        else:
            return False

    def terminate(self):
        obj_list = [] #self.pr2.scene.get_known_object_names_in_roi(minx=0.0, miny=-0.3, minz=0.3, maxx=1.2, maxy=0.3, maxz=1.5)
        print('Objects within range: ', obj_list)
        if len(obj_list) > 0:
            return True
        else:
            return False

    def act_once(self):
        print('TODO!!')
        pass

    def act(self):
        while self.current_action < len(self.action_sequence_p):
            if len(self.action_sequence_p[self.current_action]) > 1:
                print('This is a move command', self.current_action)
                success = self.pr2_left.move_to_p_rpy(self.action_sequence_p[self.current_action], 
                                                 self.action_sequence_rpy[self.current_action])
                if (not success):
                    print('Couldn\'t move to that pose')
                    return False
            else:
                print('PR2 Command', self.action_sequence_p[self.current_action])
                f = getattr(self.pr2_left, self.action_sequence_p[self.current_action][0])
                f(self.action_sequence_rpy[self.current_action][0])

            self.current_action += 1
            if self.terminate():
                print('Had to terminate!')
                return False

        print('Whole sequence done')
        return True

    def reset(self):
        self.current_action = 0

    def future_step(self):
        print('Future step.')
        offset = 1
        while self.current_action + offset < len(self.action_sequence_p):
            if len(self.action_sequence_p[self.current_action + offset]) > 1:
                print('This is a move command', self.current_action+offset)
                plan = self.pr2_left.plan(self.action_sequence_p[self.current_action + offset], 
                                                 self.action_sequence_rpy[self.current_action + offset])
                print('Plan: ', plan)
                print('Last joint states: ', plan.joint_trajectory.points[-1].positions)
                return
            else:
                offset +=1
        return 'Current state, as it is the last in this option.'


def main():
    o1 = PickupGear()
    o1.pr2_left.service_left_fk([0]*7)
    o1.act()
    # if (o1.in_initiation_set()):
    #     o1.future_step()
    # # o1.act()



if __name__ == '__main__':
    main()
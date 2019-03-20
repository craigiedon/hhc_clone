import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import robotcontroller as rc
import math

class PickupGear(object):
    """docstring for Option1: Grab gear"""
    def __init__(self, pr2_left=None, pr2_right=None):
        super(PickupGear, self).__init__()
        if pr2_left is None:
            self.pr2 = rc.PR2RobotController('left_arm', add_table=True)
        else:
            self.pr2 = pr2_left

        # self.pr2.reset_pose()


        self.action_sequence_p = [
                                 ['remove_and_deattach_gears'],
                                 ['close_gripper'],
                                 [0.20, 0.7, 0.88],
                                 [0.50, 0.7, 0.92],
                                 [0.50, 0.7, 0.88],
                                 [0.44, 0.7, 0.88],
                                 [0.44, 0.7, 0.92],
                                 [0.25, 0.7, 0.92],
                                 [0.25, 0.7, 0.73],
                                 ['open_gripper'],
                                 [0.30, 0.7, 0.73],
                                 ['close_gripper'],
                                 ['add_and_attach_gears'],
                                 [0.20, 0.7, 0.73],
                                 ]
        self.action_sequence_rpy = [['gear'],
                                    [10],
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
                                    ['gear'],
                                    [math.pi/2., 0, 0],
                                    ]
        self.current_action = 0


    def in_initiation_state(self):
        return True

    def terminate(self):
        obj_list = [] #self.pr2.scene.get_known_object_names_in_roi(minx=0.0, miny=-0.3, minz=0.3, maxx=1.2, maxy=0.3, maxz=1.5)
        print('Objects within range: ', obj_list)
        if len(obj_list) > 0:
            return True
        else:
            return False

    def act(self):
        while self.current_action < len(self.action_sequence_p):
            if len(self.action_sequence_p[self.current_action]) > 1:
                print('This is a move command')
                success = self.pr2.move_to_p_rpy(self.action_sequence_p[self.current_action], 
                                                 self.action_sequence_rpy[self.current_action])
                if (not success):
                    print('Couldn\'t move to that pose')
                    return False
            else:
                print('PR2 Command', self.action_sequence_p[self.current_action])
                f = getattr(self.pr2, self.action_sequence_p[self.current_action][0])
                f(self.action_sequence_rpy[self.current_action][0])

            self.current_action += 1
            if self.terminate():
                print('Had to terminate!')
                return False

        print('Whole sequence done')
        return True

    def reset(self):
        self.current_action = 0



def main():
    o1 = PickupGear()
    o1.act()



if __name__ == '__main__':
    main()
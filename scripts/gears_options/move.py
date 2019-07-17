import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import robotcontroller as rc
import tf
import math


class MoveTo(object):
    """docstring for MoveTo
    Navigate to a location that is delta pose away from the other arm
    """
    def __init__(self, delta_pose=[0.1, 0.49, 0.07], pr2_left=None, pr2_right=None):
        super(MoveTo, self).__init__()
        self.delta_pose = delta_pose

        if pr2_left is None:
            self.pr2_left = rc.PR2RobotController('left_arm', add_table=True)
        else:
            self.pr2_left = pr2_left

        if pr2_right is None:
            self.pr2_right = rc.PR2RobotController('right_arm', add_table=False)
        else:
            self.pr2_right = pr2_right


        self.current_action = 0
        self.pr2_right.add_and_attach_base_plate()
        # self.prepare()


    def in_initiation_state(self):
        return True

    def terminate(self):
        obj_list = [] #self.pr2.scene.get_known_object_names_in_roi(minx=0.0, miny=-0.3, minz=0.3, maxx=1.2, maxy=0.3, maxz=1.5)
        print('Objects within range: ', obj_list)
        if len(obj_list) > 0:
            return True
        else:
            return False

    def prepare(self):
        self.pr2_right.reset_pose()

        right = self.pr2_right.get_current_pose()
        left = self.pr2_left.get_current_pose()

        print(right)
        right.position.x += self.delta_pose[0]
        right.position.y += self.delta_pose[1]
        right.position.z += self.delta_pose[2]

        plan = self.pr2_left.plan(right)
        # print('Plan: ', plan)
        # plan.joint_trajectory.points[-1].positions

        self.action_sequence_p= [[right.position.x, right.position.y, right.position.z]]
        # q = (right.orientation.x, right.orientation.y, right.orientation.z, right.orientation.w)
        euler = [math.pi/2., 0, -math.pi/2]#tf.transformations.euler_from_quaternion(q)
        self.action_sequence_rpy=[euler]


    def act(self):
        self.prepare()

        while self.current_action < len(self.action_sequence_p):
            if len(self.action_sequence_p[self.current_action]) > 1:
                print('This is a move command')
                success = self.pr2_left.move_to_p_rpy(self.action_sequence_p[self.current_action], 
                                                 self.action_sequence_rpy[self.current_action])
                if (not success):
                    print('Couldn\'t move to that pose')
                    return False
            else:
                print('PR2_left Command', self.action_sequence_p[self.current_action])
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


def main():
    o1 = MoveTo()
    o1.act()



if __name__ == '__main__':
    main()
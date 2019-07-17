from pickup import PickupGear
from move import MoveTo
from inference_insert_gear import GearInsertPolicy

from robotcontroller import PR2RobotController
import rospy

def main():

    pr2_left = PR2RobotController('left_arm', add_table=True)
    pr2_right = PR2RobotController('right_arm', add_table=False)


    o1 = PickupGear(pr2_left=pr2_left, pr2_right=pr2_right)
    o2 = MoveTo(pr2_left=pr2_left, pr2_right=pr2_right)
    o3 = GearInsertPolicy(pr2_left=pr2_left, pr2_right=pr2_right)


    o1.act()
    o2.act()
    

    r = rospy.Rate(0.5)
    counter = 0

    while not rospy.is_shutdown() and counter < 100:
        print('Check some things...')
        o3.act()
        r.sleep()
        counter += 1
        print('Attempt', counter)


    print('Done.')


if __name__ == '__main__':
    main()
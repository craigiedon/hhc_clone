if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please specify torso height. Use range [0, 0.35]."
    exit 1
fi

rostopic pub --once /torso_controller/joint_trajectory_action/goal pr2_controllers_msgs/JointTrajectoryActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  trajectory:
    header:
      seq: 0
      stamp:
        secs: 0
        nsecs: 0
      frame_id: ''
    joint_names:
    - 'torso_lift_joint'
    points:
    - positions: [$1]
      velocities: [1]
      accelerations: [1]
      effort: [1]
      time_from_start: {secs: 0, nsecs: 1000000}"


if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please specify $1 - bag_name i.e. test1.bag and $2 record duration i.e. [1, 36]s. It is usually 12 sec per cube."
    exit 1
fi

ssh pr2admin@primec1 /home/pr2admin/catkin_ws/devel/lib/pr2_picknplace/pr2_picknplace_actest 6 &

rosbag record --duration=$2 -o $1 \
				  /r_forearm_cam/image_color \
				  /l_forearm_cam/image_color \
				  /narrow_stereo/left/image_rect_color \
				  /wide_stereo/left/image_rect_color \
				  /kinect2/qhd/image_color_rect \
				  /tf \
				  /tf_static \
				  /joint_states \
				  /clock \
				  /accelerometer/l_gripper_motor \
				  /accelerometer/r_gripper_motor \
				  /pressure/l_gripper_motor \
				  /pressure/r_gripper_motor \
				  /torso_lift_imu/data

#				  /netcam_stream_117/image_color \
 
#/kinect2/sd/points




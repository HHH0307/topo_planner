#!/bin/bash
echo "************************** 启动仿真环境 **************************"
source install/setup.bash
ros2 launch vehicle_simulator system_common.launch.py

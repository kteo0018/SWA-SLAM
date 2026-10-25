#!/bin/bash
set -e
cd ~/swa_portfolio
rm -f DONE
ffmpeg -y -loglevel error -framerate 60 -pattern_type glob \
  -i '/home/ken/ORB-SLAM3-ROS2-Docker-Fork/s1_gz_dataset/gz_dyna_palm_p1r1_tum/rgb/*.png' \
  -c:v libx264 -pix_fmt yuv420p -crf 23 01_camera_feed.mp4
echo "video1 done"
python3 trunk_overlay.py
echo "video2 done"
python3 traj_anim.py
echo "video3 done"
ls -la *.mp4 > DONE

# Installation

## Hardware Installation
For hardware setup, please refer to the official Droid documentation:
[Hardware Setup Guide](https://droid-dataset.github.io/droid/docs/hardware-setup)

## Software Installation
- The software installation includes four steps:
  1. Configuring the Franka robot
  2. Configuring the Oculus Quest
  3. Configuring the NUC
  4. Configuring the Laptop/Workstation

- The first three steps follow the instructions provided in the official Droid setup guide:
[Host Installation Guide](https://droid-dataset.github.io/droid/software-setup/host-installation.html)

- The only difference is in the fourth step of the guide. Firstly boot with Ubuntu 22.04 and configure static IP address which still follow the instruction of Droid above. You also need to install ZED SDK and ZED python API following [this link](https://www.stereolabs.com/docs/installation/linux). Then run the following commands:
```bash
  git clone git@github.com:horipse/fp3.git
  cd fp3
  git submodule sync
  git submodule update --init --recursive
  conda create -n "robot" python=3.7
  conda activate robot
  sudo apt update
  sudo apt install build-essential
  pip install -e ./droid/oculus_reader
  sudo apt install android-tools-adb
  pip install -e ./droid_policy_learning
  pip install -e .
  pip install -r requirement.txt
  pip install dm-robotics-moma==0.5.0 --no-deps
  pip install dm-robotics-transformations==0.5.0 --no-deps
  pip install dm-robotics-agentflow==0.5.0 --no-deps
  pip install dm-robotics-geometry==0.5.0 --no-deps
  pip install dm-robotics-manipulation==0.5.0 --no-deps
  pip install dm-robotics-controllers==0.5.0 --no-deps
```

- After installation, some checkpoints should be downloaded for future using. Download [Uni3D Model](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-l/model.pt) into `droid_policy_learning/Uni3D_large/`.

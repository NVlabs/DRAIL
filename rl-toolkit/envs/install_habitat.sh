# Run with `source envs/install_habitat.sh`
# Habitat Sim
#conda create -n hab python=3.6 cmake=3.14.0
#source activate hab
git clone --branch articulated-objects-prototype https://github.com/facebookresearch/habitat-sim.git ~/habitat-sim
cd ~/habitat-sim
pip install -r requirements.txt
#sudo apt-get install -y --no-install-recommends \
#     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
python setup.py install --with-cuda --headless --bullet
## Install Habitat API
#git clone --branch stable https://github.com/facebookresearch/habitat-api.git ~/habitat-api
#cd ~/habitat-api
#pip install -e .

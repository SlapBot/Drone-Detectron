# Drone Detection

Detection of drones using FastRCNN via [CNTK's implemententation](https://github.com/Microsoft/CNTK/wiki/Object-Detection-using-Fast-R-CNN)

### Bounding box detection of drones (small scale quadcopters)

![alt text](https://i.imgur.com/8by9aIh.gif "Detection results")

# Installation Instructions:

## Anaconda
(Install Anaconda previously or let the install-cntk.sh do it for you, recommended to do it previously)

https://www.anaconda.com/download/#linux

## CNTK (linux)
Download CNTK-2.3.1 (CPU only) tar.gz = https://cntk.ai/dllc-2.3.html
Extract it out.

`cd /home/username/cntk/Scripts/install/linux`

`./install-cntk.sh --py-version 34`

Above command will make an env called cntk-py34

Activate above env by command: `source activate cntk-py34`

Activate CNTK env by command: `source "/home/slapbot/cntk/activate-cntk"`

## Repository
Clone the repo: `git clone https://github.com/SlapBot/Drone-Detectron.git`

Cd in: `cd drone-detection/Detection/FastRCNN`

## Install AlexNet Model
Install the AlexNet Model: `python install_fastrcnn.py`

## Install Python Package Deps
Install its deps: `pip install -r requirements.txt`

Install Scikit-Image: `pip install scikit-image`

## Reinstall Broken Deps
Remove pre-installed opencv from conda (bug) and install latest one from pip via commands:

`conda remove opencv`

`pip install opencv-python`

# Testing Instructions

## PreProcess Data
Run Selective Search by command: `python A1_GenerateInputROIs.py`

Visualize the selective search results: `python B1_VisualizeInputROIs.py`

Check recall of proposed regions found in selective search task: `python B2_EvaluateInputROIs.py`

## Training Model
Finally train the model using:

`cntk configFile=/home/username/drone-detection/Detection/FastRCNN/proc/Drones_500/cntkFiles/fastrcnn.cntk currentDirectory=/home/username/drone-detection/Detection/FastRCNN/proc/Drones_500/cntkFiles/ NumLabels=3 NumTrainROIs=500 TrainROIDim=2000 TrainROILabelDim=1500 NumTestROIs=500 TestROIDim=2000 TestROILabelDim=1500`

# Evaluating and Visualizing Model
Evaluate and See the results using: `python evaluateDetections.py`


Change the value count to any value upto which you wanna see the results at: 

drone-detection/Detection/FastRCNN/imdb_data.py L:236 `visualize_multiple(visualizers[2:], count=10)`

## Credits

https://github.com/creiser/drone-detection

Used creiser's original repository and made various dependency and code changes to get it back to working.


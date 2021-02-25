## Introduction
This repo consists of code accompanying "AutoFuzz: Grammar-Based Fuzzing for Self-Driving Car Controller". AutoFuzz is a grammar-based input fuzzing tool for end-to-end self-driving car controllers. It analyzes CARLA specification to generate semantically and temporally valid test scenario with support of multiple search methods.


## Setup
### Requirements
* OS: Ubuntu 18.04
* CPU: at least 8 cores
* GPU: at least 8GB memory
* Python 3.7
* Carla 0.9.9.4 (for installation details, see below)


### Cloning this Repository

Clone this repo with all its submodules

```
git clone https://github.com/AIasd/2020_CARLA_challenge.git --recursive
```
### Create Conda Environment and Install Python Packages
All python packages used are specified in `environment.yml`.

With conda installed, create the conda environment and install python packages used:
```
conda env create -f environment.yml
```
A conda environment with name `carla99` should be created.

Activate this environment by running:
```
conda activate carla99
```


### Installation of Carla 0.9.9.4
This code uses CARLA 0.9.9.4. You will need to first install CARLA 0.9.9.4, along with the additional maps.
See [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9) for more instructions.

For convenience, the following commands can be used to install carla 0.9.9.4.

Download CARLA_0.9.9.4.tar.gz and AdditionalMaps_0.9.9.4.tar.gz from [link](https://github.com/carla-simulator/carla/releases/tag/0.9.9), put it at the same level of this repo, and run
```
mkdir carla_0994_no_rss
tar -xvzf CARLA_0.9.9.4.tar.gz -C carla_0994_no_rss
```
move `AdditionalMaps_0.9.9.4.tar.gz` to `carla_0994_no_rss/Import/` and in the folder `carla_0994_no_rss/` run:
```
./ImportAssets.sh
```
Then, run
```
cd carla_0994_no_rss/PythonAPI/carla/dist
easy_install carla-0.9.9-py3.7-linux-x86_64.egg
```
Test the installation by running
```
cd ../../..
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```
A window should pop up.

### Download a LBC pretrained model
LBC model is one of the models supported to be tested. A pretrained-model's checkpoint can be found at LBC author's provided [Wandb project](https://app.wandb.ai/bradyz/2020_carla_challenge_lbc).

Navigate to one of the runs, like https://app.wandb.ai/bradyz/2020_carla_challenge_lbc/runs/command_coefficient=0.01_sample_by=even_stage2/files

Go to the "files" tab, and download the model weights, named "epoch=24.ckpt", and pass in the file path as the `TEAM_CONFIG` in `run_agent.sh`. Move this model's checkpoint to the `models` folder (May need to create `models` folder under this repo's folder).



## Run Fuzzing
```
python ga_fuzzing.py -p 2015 -s 8791 -d 8792 --n_gen 6 --pop_size 50 -r 'town05_right_0' -c 'leading_car_braking_town05_fixed_npc_num' --algorithm_name nsga2-un --has_run_num 300 --objective_weights -1 1 1 0 0 0 0 0 0 0 --check_unique_coeff 0 0.2 0.5
```
For more API information, checkout the interface inside `ga_fuzzing.py`.





## Check out maps and find coordinates
Check out the map details by spinning up a CARLA server

```
./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600 -opengl
```
and running
```
python inspect_routes.py
```
Also see the corresponding birdview layout [here](https://carla.readthedocs.io/en/latest/core_map/) for direction and traffic lights information.

Note to switch town map, one can change the corresponding variable inside this script.

## Extract short routes extracted from CARLA challenge routes
```
python generate_short_routes.py
```

## Run on short routes
```
python run_new_routes.py
```


## Retrain model from scratch
Note: the retraining code only supports single-GPU training.
Download dataset [here](https://drive.google.com/file/d/1dwt9_EvXB1a6ihlMVMyYx0Bw0mN27SLy/view). Add the extra data got from fuzzing into the folder of the dataset and then run stage 1 and stage 2.

Stage 1 (~24 hrs on 2080Ti):
```
CUDA_VISIBLE_DEVICES=0 python carla_project/src/map_model.py --dataset_dir path/to/data
```

Stage 2 (~36 hrs on 2080Ti):
```
CUDA_VISIBLE_DEVICES=0 python carla_project/src/image_model.py --dataset_dir path/to/data --teacher_path path/to/model/from/stage1
```

## Model Fixing


Stage 2 finetuning:
```
CUDA_VISIBLE_DEVICES=0 python carla_project/src/image_model.py --dataset_dir path/to/data --teacher_path path/to/model/from/stage1
```



# Reference
This repo is partially built on top of [Carla Challenge (with LBC supported)](https://github.com/bradyz/2020_CARLA_challenge) and [pymoo](https://github.com/msu-coinlab/pymoo)

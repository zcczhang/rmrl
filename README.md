# When Learning Is Out of Reach, Reset: Generalization in Autonomous Visuomotor Reinforcement Learning

**[[ArXiv]](https://arxiv.org/abs/2303.17600) [[PDF]](https://zcczhang.github.io/rmrl/assets/paper.pdf) [[project page]](https://zcczhang.github.io/rmrl)**

[Zichen "Charles" Zhang](https://zcczhang.github.io/), [Luca Weihs](https://lucaweihs.github.io)

PRIOR @ Allen Institute for AI

![](https://zcczhang.github.io/rmrl/assets/images/teaser.jpg)


## Citation
If you find this project useful in your research, please consider citing:
```
@article{zhang2023when,
  title   = {When Learning Is Out of Reach, Reset: Generalization in Autonomous Visuomotor Reinforcement Learning},
  author  = {Zichen Zhang and Luca Weihs},
  year    = {2023},
  journal = {arXiv preprint arXiv: Arxiv-2303.17600},
}
```

## Installation
- Follow the [instruction](https://github.com/openai/mujoco-py#install-mujoco) for installing `mujuco-py` and install the following apt packages if using Ubuntu:
```commandline
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```
- Clone this repository locally
```commandline
git clone https://github.com/zcczhang/rmrl.git && cd rmrl
pip install -e .
```
- Alternatively, create conda environment with name `rmrl`
```commandline
conda env create --file ./conda/environment-base.yml --name rmrl
conda activate rmrl
```

[//]: # (The above is very simple but has the side effect of creating a new `src` directory where it will place some of repository's dependencies. To get around this, instead of running the above you can instead run the commands:)

[//]: # (```commandline)

[//]: # (export MY_ENV_NAME=rmrl)

[//]: # (export CONDA_BASE="$&#40;dirname $&#40;dirname "${CONDA_EXE}"&#41;&#41;")

[//]: # (export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc")

[//]: # (conda env create --file ./conda/environment-base.yml --name $MY_ENV_NAME)

[//]: # (```)

The <span style="font-variant: small-caps;font-size: 110%">Stretch-P&P</span> and RoboTHOR ObjectNav is built based on [AI2-THOR](https://github.com/allenai/ai2thor). After installing the requirements, you could start the xserver by running
```commandline
python scripts/startx.py
```

## Simulations

https://user-images.githubusercontent.com/52727818/229280231-03b88fc1-36c2-4d9f-9426-d819f8383792.mp4

<br>

### <span style="font-variant: small-caps;font-size: 130%"><i>Stretch-P&P</i></span>

Example usage can be found at jupyter notebook [here](scripts/Stretch-P&P.ipynb). APIs are following as [iTHOR](https://ai2thor.allenai.org/ithor/documentation) and [ManipulaTHOR](https://github.com/allenai/manipulathor#-training-an-agent). Controller parameters and other constants can be found at [here](allenact_plugins/stretch_manipulathor_plugin/stretch_constants.py) (e.g. object partitions, action scales, e.t.c.). To modify the scene and objects (in Unity), see instructions [here](https://github.com/allenai/ai2thor/tree/main/unity). Now we provide details about the benchmark:


#### Sensory Observations
The types of [sensors](allenact_plugins/stretch_manipulathor_plugin/strech_sensors.py) provided for this task include:
1. **RGB image** (`RGBSensorStretch`) - `224x224x3` egocentric camera mounted at the agent wrist. 
2. **egomotion** (`StretchPolarSensor`) - `2+2=4` dimensional agent gripper position and target goal position relatively to the agent base.
3. **prompt** (`StretchPromptSensor`) - language prompt including the picking object and target obect/egocentric point goal. E.g `Put red apple to stripe plate.`


#### Action Space
A total of 10 actions are available to our agents, these include (x, y, z are relative to robot base):


| **Action**          | **Description**                                                                                                   | **Scale** |
|---------------------|-------------------------------------------------------------------------------------------------------------------|-----------|
| `MoveAhead`         | Move robot base in $+x$ axis                                                                                      | 5 cm      |
| `MoveBack`          | Move robot base in $-x$ axis                                                                                      | 5 cm      |
| `MoveArmHeightP`    | Increase the arm height ($+y$)                                                                                    | 5 cm      |
| `MoveArmHeightM`    | Decrease the arm height ($-y$)                                                                                    | 5 cm      |
| `MoveArmP`          | Extend the arm horizontally ($+z$)                                                                                | 5 cm      |
| `MoveArmM`          | Retract the arm horizontally ($-z$)                                                                               | 5 cm      |
| `MoveWristP`        | Rotate the gripper in $+x$ yaw direction                                                                          | $2^\circ$ |
| `MoveWristM`        | Rotate the gripper in $-x$ yaw direction                                                                          | $2^\circ$ |
| `PickUp(object_id)` | Pick up object with specified unique  `object_id` if object within the sphere with radius $r$ centered at gripper | $r=0.06$  |
| `Release`           | Release object with simulation steps until object is relatively stable                                            | --        |


[//]: # (1. moving the agent base)

[//]: # (   - `MoveAhead` move robot base in +x axis 5cm at most)

[//]: # (   - `MoveBack` move robot base in -x axis 5cm at most)

[//]: # (2. moving the arm)

[//]: # (    - `MoveArmHeightP` increase the arm height &#40;+y&#41; 5cm at most)

[//]: # (    - `MoveArmHeightM` decrease the arm height &#40;-y&#41; 5cm at most)

[//]: # (    - `MoveArmZP` extend the arm horizontally &#40;+z&#41; 5cm at most)

[//]: # (    - `MoveArmZP` retract the arm horizontally &#40;-z&#41; 5cm at most)

[//]: # (3. rotate the gripper)

[//]: # (    - `MoveWristP` rotate the gripper in +x yaw direction $2^\circ$ at most)

[//]: # (    - `MoveWristM` rotate the gripper in -x yaw direction $2^\circ$ at most)

[//]: # (4. Abstract Pick & Place)

[//]: # (    - `PickUp&#40;object_id&#41;` Pick up object with specified unique object id if object within)

[//]: # (     the sphere with radius r=0.06 centered at gripper)

[//]: # (    - `ReleaseObject` Release object &#40;if holding at hand&#41; with simulation steps until object is relatively stable)

In order to define a new task or change any component of the training procedure, it suffices to look into/change the following files and classes.

#### Environment

The `StretchManipulaTHOREnvironment` defined in `allenact_plugins/stretch_manipulathor_plugin/stretch_arm_environment.py` is a wrapper around the AI2-THOR environment which helps with discretizing the action space for the pick-and-place tasks and wraps the functions that are needed for working with the low-level manipulation features.

<details><summary>Important Features</summary>
<p>

- `step(action)` translates the `action` generated by the model to their corresponding AI2THOR API commands.
- `is_object_at_low_level_hand(object_id)` Checks whether the object with unique id `object_id` is at hand or not.
- `get_absolute_hand_state` get the position, rotation, and hand radius for current state
- `get_object_by_id(object_id)` get metadata of a specified object
- `teleport_object(target_object_id, target_position, rotation, fixed)` helper function that teleports the specified object to a target position and rotation, and set stationary if `fixed`.
- `randomize_texture_lightning` deterministic texture and lightning randomization function that randomly sample or change from specified for table, table leg, wall, floor, and/or light's cosmetic augmentations.
- `scale_object(object_id, scale)` wrapped function that scale the selected object
- `hand_reachable_space_on_countertop(countertop_id)` get a proximity xyz limits that the gripper can reach considering the specified countertop.

</p>
</details>

#### P&P Task

`StretchPickPlace` task class can be found at `allenact_plugins/stretch_manipulathor_plugin/stretch_tasks/strech_pick_place.py`. This class includes the possible actions, reward definition, metric calculation and recording and calling the appropriate API functions on the environment.

<details><summary>Important Features</summary>
<p>

- `success_criteria` the picking object’s bounding box should intersect with the receptacle trigger box (which is different from the bounding box and only includes the area of the receptacle, e.g. internal rectangular area of a pan without the handle) when both objects static. Secondly, the distance between the picking object and the center of the receptacle trigger box must be within a threshold to avoid edge cases or large receptacles. In the case of random targets or point goals, only the second criterion is used.
- `metrics` Calculates and logs the value of each evaluation metric per episode.
- `current_state_metadata` useful state information each step, containing agent, hand, picking object, goal state metadata.
- `task_name` prompt that parsed from current task
- `setup_from_task_data` setup initial configuration from the input [`TaskData`](allenact_plugins/stretch_manipulathor_plugin/dataset_utils.py) (including scene name, picking object metadata, receptacle metadata, etc).
- `is_obj_off_table` checking whether the picking object is off its parent receptacle

</p>
</details>

#### P&P TaskSampler

`StretchExpRoomPickPlaceTaskSampler` and`StretchExpRoomPickPlaceResetFreeTaskSampler` for episodic and RF/RM RL can be found at `allenact_plugins/stretch_manipulathor_plugin/strech_task_sampler_exproom.py`. These class is in charge of initializing the all possible locations for the object and agent and randomly sampling a data point for each episode for episodic RL or *phase* for RM-RL.

<details><summary>Important Features</summary>
<p>

- `need_hard_reset` reset criteria checked every phase/episode. Also related to `dispersion_measure_check` and `distance_measure_check` which implement our methods. 
- `sample_new_combo` sample the picking and placing objects or point goals for next task, considering the RM-RL algorithm (random targets or two-phases forward-backward), budgets (possible objects in current distributed process)
- `next_task` creates next task instance for interactions. If an intervention is determined by methods (e.g. episodic, measurement-led, periodic) gets the source and target locations, initializes the agent and transport the object to its initial state. If using random targets for RM-RL, sample a reasonable point goal here.

</p>
</details>

### <span style="font-size: 120%"><i>Sawyer Peg</i></span>

Some pre-registered gym env can be found at [here](allenact_plugins/sawyer_peg_plugin/registered_envs.py). For example, 

<details><summary>a RM-RL with std measurement-determined reset with random goal environment</summary>
<p>

```python
from allenact_plugins.sawyer_peg_plugin import *

# Examples of initializing a `Std` measure-determined intervention with random goals for training
env = gym.make("RFSawyerPegRandomStdMeasureVisual-v1")
# or
env = parse_sawyer_peg_env("visual_std_measure")
```
</p>
</details>

<details><summary>an episodic environment with random peg box and hole positions for evaluation can be made by</summary>
<p>

```python
# Examples of initializing an episodic evaluation environment with random peg box and hole positions
env_eval = gym.make("SawyerPegVisual-v1")
# or
env_eval = parse_sawyer_peg_env("visual_eval")
```

</p>
</details>

### <span style="font-size: 120%"><i>RoboTHOR ObjectNav</i></span>

To get the ObjectNav scenes dataset for `RoboTHOR` run the following command:

```commandline
bash datasets/download_navigation_datasets.sh robothor-objectnav
```

This will download the dataset into `datasets/robothor-objectnav`. Full documentation can be found at [here](https://ai2thor.allenai.org/robothor/documentation). The reset-free/reset-minimized task sampler can be found at [here](allenact_plugins/robothor_plugin/robothor_task_samplers.py). For example, 

<details><summary>task sampler for a RM-RL agent with std measurement-led reset and random targets</summary>
<p>

```python
import os
import glob
import numpy as np
from allenact_plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler


# See experiments wrapper for distributed partitions
dataset_dir = "datasets/robothor-objectnav/train"
scenes_path = os.path.join(dataset_dir, "episode", "*.json.gz")
scenes = [
    scene.split("/")[-1].split(".")[0] 
    for scene in glob.glob(scenes_path)
]
task_sampler = ObjectNavDatasetTaskSampler(
    scenes=scenes,
    scene_directory=dataset_dir,
    max_steps=300,
    # False then episodic
    reset_free=True,
    # False for two-phase FB-RL
    measurement_lead_reset=True,
    # std, entropy, euclidean, dtw
    measure_method="std",
    # other periodic resets
    num_steps_for_reset=np.inf,
    
)
```

</p>
</details>

## Experiments

Experiment files are under `project` directory. In general, run below commands for training:


```commandline
python main.py \
{experiment_file} \
-b projects/{environment_name}/experiments/{base_folder} \
-o {output_path}
```

where `{experiment_file}` is python file name for the experiment, `{environment_name}` is one of [`objectnav`](projects/objectnav), [`saywer_peg`](projects/sawyer_peg), and [`strech_manipulation`](projects/strech_manipulation), `{base_folder}` is the base folder for `{experiment_file}`, and `{output_path}` is output path for saving experiment checkpoints and configurations. See possible files and directories for every experiment below.

Optional:
- `--callbacks allenact_plugins/callbacks/wandb_callback.py` use wandb logging, where specified your wandb entry and project at each yaml config file under `cfg` directory (e.g. [here](projects/strech_manipulation/experiments/budget_measure_ablation/cfg/rgb.yaml) for Stretch-P&P)
- `--valid_on_initial_weights` validation at initial weights before training
- `-a` disable saving the used config in the output directory
- `-i` disable tensorboard logging (which by default save in the output directory)
- `-d` sets CuDNN to deterministic mode
- `-s {seed}` set seed as `n`, random seeds without setting
- `-m {n}` set maximal number of sampler processes to spawn for each worker as `n`, default set in experiment config.
- `-l debug` set logger level as `debug`.


For evaluation, run:

```commandline
python main.py \
{test_file} \
-b projects/{environment_name}/experiments/tests/{test_base_folder} \
-o {output_path} \
-e \
-c {path/to/checkpoints} \
--eval
```

where `-e` for deterministic testing and `-c` for specifying the `checkpoints` folder or a single `*.pt` file for evaluation, and `--eval` for explicitly setting for running inference pipeline. Same optional extra args as above.

### <span style="font-variant: small-caps;font-size: 130%"><i>Stretch-P&P</i></span>

To reproduce the training with our measurement-determined reset with random goals with budget $=1$ (i.e. $\texttt{Put red apple into stripe plate.}$) and <span style="font-variant: small-caps;font-size: 100%"><i>Std</i></span> metric, run

```commandline
python main.py \
random_irr_single_combo \
-b projects/strech_manipulation/experiments/budget_measure_ablation \
-o {output_path}
```

To run evaluation for <span style="font-variant: small-caps;font-size: 100%"><i>Pos-OoD</i></span>, run

```commandline
python main.py \
measure_random_novel_pos \
-b projects/strech_manipulation/experiments/tests/budget_and_measurement \
-o {output_path} \
-e \
-c {path/to/checkpoints} \
--eval
```

And the `{test_file}` can be `measure_random_novel_texture_lightning` for <span style="font-variant: small-caps;font-size: 100%"><i>Vis-OoD</i></span>; `measure_random_novel_objects` for <span style="font-variant: small-caps;font-size: 100%"><i>Obj-OoD</i></span>; and `measure_random_novel_scene` for <span style="font-variant: small-caps;font-size: 100%"><i>All-OoD</i></span>. You can change the global variable `N_COMBO` and `MEASURE` for measure name for ablating [here](projects/strech_manipulation/experiments/tests/budegt_and_measurement/measure_random_novel_pos.py#L12).

### <span style="font-size: 120%"><i>Sawyer Peg</i></span>

To reproduce the training with our measurement-determined reset with <span style="font-variant: small-caps;font-size: 100%"><i>Std</i></span> metric, run

```commandline
python main.py \
visual_std \
-b projects/sawyer_peg/experiments/measure_variations \
-o {output_path}
```

and replace `std` with `dtw`, `entropy`, or `euclidean` in `{experiment_file}` for other measurements we proposed.

To run evaluation for novel peg box and hole positions, run

```commandline
python main.py \
visual_test_novel_box \
-b projects/strech_manipulation/experiments/tests/std \
-o {output_path} \
-e \
-c {path/to/checkpoints} \
--eval
```

and replace `novel_box` with `in_domain` for in domain evaluation, and `small_table` for evaluating with narrower table.

### <span style="font-size: 120%"><i>RoboTHOR ObjectNav</i></span>

To reproduce the training with our measurement-determined reset with <span style="font-variant: small-caps;font-size: 100%"><i>Std</i></span> metric, run

```commandline
python main.py \
std \
-b projects/obejctnav/experiments/rmrl \
-o {output_path}
```
for evaluation in RoboTHOR validation scene dataset, run

```commandline
python main.py \
std \
-b projects/obejctnav/experiments/rmrl \
-o {output_path} \
-e \
-c {path/to/checkpoints} \
--eval
```

## Acknowledgement

The codebase framework is based on [AllenAct](https://allenact.org/) framework. The _Stretch-P&P_ and _RoboTHOR ObjectNav_ simulated environments are built based on [AI2-THOR](https://github.com/allenai/ai2thor). The Sawyer Peg simulation is modified from [MetaWorld](https://meta-world.github.io/).

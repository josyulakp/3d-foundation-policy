# FP3: A 3D Foundation Policy for Robotic Manipulation

 **3D Foundation Policy (FP3)** is a 1.3B 3D point cloud-based language-visuomotor policy pre-trained on 60k episodes from the DROID dataset. FP3 supports data-efficient fine-tuning for downstream tasks, while demonstrating superior generalizability to unseen environments and novel objects.

[**[Homepage]**](https://3d-foundation-policy.github.io/) &ensp; [**[Paper]**](https://arxiv.org/pdf/2503.08950)

![](assets/concept.jpg)

---------
## Installation

See [INSTALL.md](INSTALL.md) for details.

## Workflow

### 1. Data Collection
- Calibrate cameras using the Droid GUI, following the guide here: [Calibrating Cameras](https://droid-dataset.github.io/droid/example-workflows/calibrating-cameras.html). Since we use point clouds as input, ensuring proper camera calibration is crucial.
- Collect data using the Droid GUI, following the instructions here: [Data Collection](https://droid-dataset.github.io/droid/example-workflows/data-collection.html).

### 2. Data processing
- The collected image information is stored in `.svo` files. We need to further process and combine this information into `.h5` files. Navigate to the project folder and run the following command:
  ```bash
  python droid_policy_learning/robomimic/scripts/conversion/convert_droid.py --folder <your_data_folder>
  ```

- After that, run this command to gather all your data and create a `dataset.json` file:
  ```bash
  python droid_policy_learning/robomimic/scripts/conversion/set_manifest_file.py --folder <your_data_folder> --lang <language_insruction>
  ```

- With `dataset.json`, we can then encode the language with the pretrained encoder and merge the feature into `.h5` files. Run the following command:
  ```bash
  python droid_policy_learning/robomimic/scripts/conversion/add_lang_to_converted_data.py --manifest_file droid_policy_learning/dataset.json
  ```

### 3. Model Training
- Our training scripts using deepspeed to accelerate training, so you need to configure it properly.

- Add language embedding to dataset by running:
  ```bash
  python droid_policy_learning/robomimic/scripts/conversion/add_lang_to_converted_data.py --manifest_file droid_policy_learning/dataset.json
  ```

- Generate config by running:
  ```bash
  python droid_policy_learning/robomimic/scripts/generate_config.py --dataset <your manifest file> --exp <experiment name>
  ```

- And then run:
  ```bash
  accelerate launch droid_policy_learning/robomimic/scripts/train.py --config <config path> --ckpt <ckpt_path>
  ``` 

  If you want to finetune on our pretrained model, just download the checkpoint and specify the `ckpt` argument. 

### 4. Evaluation
- To evaluate a policy on a robot, you can just run:
  ```bash
  cat <your_language_instruction> > ./eval_params/lang_command.txt
  python droid/scripts/evaluation/evaluate_policy.py
  ```

---------

## Acknowledgment
Our code is generally built upon [DROID](https://github.com/droid-dataset/droid), [Robomimic](https://github.com/ARISE-Initiative/robomimic). We sincerely appreciate their contribution to the open-source community, which have significantly supported this project.  

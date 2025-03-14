# Workflow
This section provides a complete overview of the workflow for using this method.

## 1. Data Collection
- Calibrate cameras using the Droid GUI, following the guide here: [Calibrating Cameras](https://droid-dataset.github.io/droid/example-workflows/calibrating-cameras.html). Since we use point clouds as input, ensuring proper camera calibration is crucial.
- Collect data using the Droid GUI, following the instructions here: [Data Collection](https://droid-dataset.github.io/droid/example-workflows/data-collection.html).

## 2. Data processing
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

## 3. Model Training
- Our training scripts using deepspeed to accelerate training, so you need to configure it properly.

- Generate config by running:
  ```bash
  python droid_policy_learning/robomimic/scripts/conversion/add_lang_to_converted_data.py --manifest_file droid_policy_learning/dataset.json
  ```

- And then run:
  ```bash
  accelerate launch droid_policy_learning/robomimic/scripts/train.py --name <exp_name> --ckpt <ckpt_path>
  ``` 

  If you want to finetune on our pretrained model, just download the checkpoint and specify the `ckpt` argument. 

## 4. Evaluation
- To evaluate a policy on a robot, you can just run:
  ```bash
  cat <your_language_instruction> > ./eval_params/lang_command.txt
  python droid/scripts/evaluation/evaluate_policy.py
  ```


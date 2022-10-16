
## Overview
This is the code repository for my master thesis at ETH Zurich titled "Uncertainty Quantification for Image-based Traffic Prediction", submitted in Sept. 2022. You can reach out via mail at 'alex dot timans at hotmail dot com' in case of questions.

Supervisors: Dr. Lukas Meier [[LM]](https://stat.ethz.ch/~meier/), Prof. Dr. Martin Raubal [[MR]](https://www.raubal.ethz.ch), direct supervision by Ye Hong [[YH]](https://ikg.ethz.ch/en/people/staff-a-z/person-detail.MjUzMDAx.TGlzdC8xMDMxLC05MDgxNDI5Mg==.html), Nina Wiedemann [[NW]](https://baug.ethz.ch/departement/personen/mitarbeiter/personen-detail.MjUzNzEy.TGlzdC82NzksLTU1NTc1NDEwMQ==.html) and Nishant Kumar [[NK]](https://frs.ethz.ch/people/researchers/NishantKUMAR.html).

### Abstract :memo:
---
Despite the strong predictive performance of deep learning models for traffic prediction, their widespread deployment in real-world intelligent transportation systems has been restricted by a lack of interpretability and perceived trustworthiness. Uncertainty quantification (UQ) methods provide an approach to induce probabilistic reasoning, improve decision-making and enhance model deployment potential. This study investigates the application of different UQ methods for short-term traffic prediction on the image-based *Traffic4cast* dataset. We compare two epistemic and two aleatoric UQ methods on both temporal and spatio-temporal transfer tasks, and find that meaningful uncertainty estimates can be recovered. Methods are compared in terms of uncertainty calibration and sharpness, and our experiments suggest that modelling both epistemic and aleatoric uncertainty jointly produces the most accurate uncertainty estimates. Obtained uncertainty estimates are spatially related to the city-wide road network, and subsequently employed for unsupervised outlier detection on changes in city traffic dynamics. We find that our approach can capture both temporal and spatial effects on traffic behaviour, and that their interaction is complex. Our work presents a further step towards boosting uncertainty awareness in traffic prediction tasks, and aims to showcase the potential value contribution of UQ methods to the development of intelligent transportation systems, and to a better understanding of city traffic dynamics. 

---

## Folder structure
The folder structure is quite self explanatory, but here are some comments on each folder or file on the main repository level:
```
t4c2021-uncertainty-thesis/
├── data: contains the custom dataset and a script to generate the correct data folder structure
├── metrics: contains files to compute MSE, calibration and prediction interval metrics, as well as aggregate .txt results files into .csv tables
├── misc: nothing particularly interesting, last layer correlation and sandboxing
├── model: contains U-Net model architecture, parameter config, training, and checkpointing files.
├── plotting: contains scripts to visualise spatial uncertainty and metrics maps, as well as outlier detection results, and some other plots
├── uq: contains the implementations of evaluated UQ methods, as well as the test set and outlier detection evaluation scripts to produce results. This is the most important folder.
├── util: contains utility functions such as read/write data, set seed or get device. The most important file in here is 'h5_util.py'.
├── env_current.yml: The package environment file as used by myself.
├── env_t4c.yml: The package environment file as given by the *Traffic4cast* 2021 challenge organizers (https://github.com/iarai/NeurIPS2021-traffic4cast)
└── main.py: The main script to run model training and inference via CLI.
```
The implemented UQ methods as mentioned in the thesis report correspond to the following names used in the code, in particular regarding argument names for parameter ```uq_method``` in ```main.py``` or names used in the inference scripts ```eval_model.py```, ```eval_tta_ensemble.py``` and ```outlier_detection.py``` :
| Thesis report | Code |
|--|--|
| TTA + Ens | combo |
| CUB | point |
| Ens | ensemble |
| MCBN | bnorm |
| TTA | tta |
| Patches | patches |

To run the code, guidelines are provided below.

### Local code run preparation

1. Clone repo
```
gh repo clone alextimans/t4c2021-uncertainty-thesis
```
- :warning: Make sure to be in the parent directory of ```t4c2021-uncertainty-thesis``` as working directory. Let's call it ```run-code```, so set working directory to ```run-code```. This directory is also the one from which to launch code runs, which is why the provided sample runs below follow the scheme ```python t4c2021-uncertainty-thesis/[script.py -args]```.

2. Set-up python env (with conda package manager)
- Either generate python env using the 
competition environment via ```env_t4c.yml``` and add potentially missing packages manually, or use ```env_current.yml``` for a more stringent but encompassing environment.
- Uncomment related lines in either ```.yml``` file in case of local machine with GPU support.
```
conda env create -f t4c2021-uncertainty-thesis/misc/env_current.yml
conda activate t4c
```
- Initialize the code repo by resetting the python path
```
cd t4c2021-uncertainty-thesis
export PYTHONPATH="$PYTHONPATH:$PWD"
cd ..
```

3. Prepare the data
- Get the data by visiting the [Traffic4cast webpage](https://www.iarai.ac.at/traffic4cast/2021-competition/challenge/) and following instructions on accessing the data. **Note:** this requires registration with the competition website.
- Put all the data in a folder ```run-code/data/raw``` in the working directory in uncompressed city folders as given by the *Traffic4cast* competition.
- Run the following call in CLI to create data folders in line with the data structure used in this work and remove 2020 leap year days.
```
  python t4c2021-uncertainty-thesis/data/set_data_folder_str.py --data_raw_path="./data/raw" --remove_leap_days=True
```
- The data should now be separated into ```train```, ```val``` and ```test``` folders within ```run-code/data/raw/[city]```.

4. Ready! Execute desired code runs as listed below. 
5. **Note:** These commands are all for running code on your local machine. This may not be particularly suitable for e.g. model training or inference on large portions of the data, since runtimes will be extremely long. Most of the heavy computations for this work were executed on GPUs on a computing cluster. The provided runs are only sample runs to visualize the use of commands and arguments. To fully replicate results please check the details provided in the thesis report and adjust parameter values accordingly.

### Local code runs

- To check the full list of main script CLI arguments that can be given and their default values and short explanations.
```
python t4c2021-uncertainty-thesis/main.py -h
```

- Sample run: training from scratch only
```
python t4c2021-uncertainty-thesis/main.py --model_str=unet --model_id=1 --model_evaluation=False --batch_size=8 --epochs=2 --data_limit=10 --device=cpu --random_seed=9876543
```

- Sample run: continue training only from previous checkpoint
```
python t4c2021-uncertainty-thesis/main.py --model_str=unet --model_id=1 --model_evaluation=False --resume_checkpoint="[checkpoint .pt path]" --batch_size=8 --epochs=2 --data_limit=10 --device=cpu --random_seed=9876543
```

- Sample run: calibration run from previous checkpoint only. This is used to generate quantiles which are needed for a test set run. It is important to set ```batch_size=1```. ```quantiles_path``` defines where quantile files will be written to. Select the desired UQ method via ```uq_method```.
```
python t4c2021-uncertainty-thesis/main.py --model_str=unet --model_id=1 --model_training=False --model_evaluation=False --resume_checkpoint="[checkpoint .pt path]" --batch_size=1 --device=cpu --calibration=True --calibration_size=100 --uq_method=tta --quantiles_path="[quantiles path]" --random_seed=9876543 
```

- Sample run: test set evaluation run from previous checkpoint only. Again it is important to keep ```batch_size=1```. We also require quantile files that are located in the folder denoted by ```quantiles_path```, as well as a test sample indices file located at ```test_samp_path``` and called by ```test_samp_name```. The result files are written into the folder denoted by ```test_pred_path```.  Select the desired UQ method via ```uq_method```. **Note:** to run ```uq_method=combo``` properly one needs to uncomment an import statement in ```main.py``` such that ```eval_tta_ensemble.py``` is called instead of ```eval_model```.
```
python t4c2021-uncertainty-thesis/main.py --model_str=unet --model_id=1 --model_training=False --model_evaluation=True --resume_checkpoint="[checkpoint .pt path]" --batch_size=1 --device=cpu --calibration=False --uq_method=tta --quantiles_path="[quantiles path]" --test_pred_path="[results path]" --test_samp_path="[test sample indices path]" --test_samp_name="[test sample indices file name]" --random_seed=9876543
```

- Sample run: outlier detection script using epistemic uncertainty estimates via deep ensembles. There are a number of relevant boolean flags in the script which activate and deactivate certain sections, and whose meaning can be checked by calling ```python t4c2021-uncertainty-thesis/uq/outlier_detection.py -h```. Key parameters include ```fix_samp_idx```, which specifies the fixed time slot per day that we evaluate outliers on; ```cities```, which specifies the cities to run outlier detection for; and ```out_bound```, which specifies the fixed outlier bound used to decide on binary outlier labels. 
```
python t4c2021-uncertainty-thesis/uq/outlier_detection.py --model_str=unet --model_id=1 --batch_size=1 --test_pred_path="[model pred path]" --uq_method=ensemble --fix_samp_idx 120 100 110 --cities BANGKOK BARCELONA MOSCOW --out_bound=0.001 --device=cpu --random_seed=9876543 --out_name="[outlier file name]"
```
- Sample run: create test set random sample indices
```
python t4c2021-uncertainty-thesis/metrics/get_random_sample_idx.py --samp_size=100 --test_samp_path="[test sample file path]" --test_samp_name="[test sample file name]"
```

- Sample run: given obtained results, aggregate into .csv table
```
python t4c2021-uncertainty-thesis/metrics/get_scores_to_table.py --test_pred_path="[results folder path]"
```

- Sample run: aggregate multiple .csv tables into one by taking means over scores
```
python t4c2021-uncertainty-thesis/metrics/get_scores_to_table.py --test_pred_path="[results folder path]" --create_table=False --agg_table=True --agg_nr=5
```

- Sample run: plot epistemic histograms with KDE fits for a fixed pixel (visual inspection of part of outlier detection framework)
```
python t4c2021-uncertainty-thesis/plotting/epistemic_hist.py --test_pred_path="[results folder path]" --uq_method=ensemble --pixel 100 100 --mask_nonzero=False --fig_path="[figure path]"
```
- To generate other (spatial) plots, one has to directly work in the respective scripts  ```uncertainty_maps.py``` and ```outlier_maps.py``` and set parameter values at the top of the script (there is no access point via CLI).

### Other code info

- Model runs perform a healthy amount of logging info to keep the code run process transparent, and also contains multiple assert statements for file path checks.

- Running the training sample run from scratch should result in a folder ```unet_1``` created in ```[parent dir]/checkpoints``` with a bunch of loss files. There are: train + val loss files per batch and per epoch. On top there should be two model checkpoint files (e.g. ```unet_ep0_05061706.pt``` and ```unet_ep1_05061707.pt``` because we trained for two epochs and save each epoch as set by the ```save_each_epoch``` params in ```t4c2021-uncertainty-thesis/model/configs.py, earlystop_config```.)

- Setting the CLI argument ```model_id``` properly is crucial as this will be used to create the checkpoint folder. If not set properly and training a new model, the files in the existing folder ```[parent dir]/checkpoints/unet_{model_id}``` may risk being overwritten and lost. Thus whenever training a new model from scratch make sure that ```model_id``` is unique to create a new checkpoint folder. When continuing training of an already existing model checkpoint, make sure to set the ```model_id``` to the **same** value so that subsequent checkpoint files are saved in the same checkpoint folder (it is not a requirement but makes sense to avoid confusion).

- To run a model on the full data (either train + val or test or both) simply omit any data limiting arguments, namely ```data_limit, train_data_limit, val_data_limit, test_data_limit```. 

- To specifically switch off either training or test set evaluation one needs to explicitly set the respective argument to ```False```, i.e. either ```model_training=False``` or ```model_evaluation=False```. Defaults for both are ```True```. ```calibration``` is set by default to ```False```.

- The device is either automatically inferred from the given machine or explicitly handed via argument ```device```. It takes either values ```cpu``` or ```cuda```. To run on multiple GPUs, set argument ```data_parallel=True``` which will activate PyTorch's DataParallel framework (if those are available). Training etc. should all work smoothly in the same manner as running on e.g. ```cpu```.

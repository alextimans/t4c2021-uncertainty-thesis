## Local run prep

1. Clone repo
```
gh repo clone alextimans/t4c2021-uncertainty-thesis
```

2. Set python environment
```
cd t4c2021-uncertainty-thesis
conda activate [local package env]
export PYTHONPATH="$PYTHONPATH:$PWD"
```

3. Prepare the data
  - Make sure that all the city data is in a folder ```t4c2021-uncertainty-thesis/data/raw``` in uncompressed city folders as given by the competition.
  - Optional: Manually remove 4 files ```2020-02-29_{city}_8ch.h5``` for leap year 2020, where ```city=[ANTWERP, BANGKOK, BARCELONA, MOSCOW]```
  - To obtain the train-val-test split run the following in CLI.
  ```
  python data/set_data_folder_str.py --data_raw_path="./data/raw"
  ```

4. Ready! Execute desired model runs.

## Local model runs

- To check the main script CLI arguments that can be given and their default values and short explanations.
```
python main.py -h
```

- Sample run: training from scratch
```
python main.py --model_str=unet --model_id=1 --model_evaluation=False --batch_size=2 --epochs=2 --data_limit=4
```

- Sample run: continue **training only** from previous checkpoint
```
python main.py --model_str=unet --model_id=1 --model_evaluation=False --resume_checkpoint="./checkpoints/unet_1/unet_ep1_05061639.pt" --batch_size=2 --epochs=2 --data_limit=4
```

- Sample run: test set evaluation from previous checkpoint
```
python main.py --model_str=unet --model_id=1 --model_training=False --resume_checkpoint="./checkpoints/unet_1/unet_ep1_05061639.pt" --batch_size=2 --test_data_limit 1 2 2
```

- Sample run: training + test set evaluation from scratch
```
python main.py --model_str=unet --model_id=1 --batch_size=2 --epochs=2 --data_limit=4 --test_data_limit 1 2 2
```

## More info

- Model runs perform a healthy amount of logging info to keep the model run process transparent.

- Running the sample run ```python main.py --model_str=unet --model_id=1 --batch_size=2 --epochs=2 --data_limit=4 --test_data_limit 1 2 2``` should result in a folder ```unet_1``` created in ```t4c2021-uncertainty-thesis/checkpoints``` with a bunch of loss files as currently shown in the repo checkpoints folder as an example. There are: train + val loss files per batch and per epoch, one test loss file. On top there should be two model checkpoint files (e.g. ```unet_ep0_05061706.pt``` and ```unet_ep1_05061707.pt``` because we trained for two epochs and save each epoch as set by the ```save_each_epoch``` params in ```t4c2021-uncertainty-thesis/model/configs.py, earlystop_config```.)

- Setting the CLI argument ```model_id``` properly is crucial as this will be used to create the checkpoint folder. If not set properly and training a new model, the files in the existing folder ```t4c2021-uncertainty-thesis/checkpoints/unet_{model_id}``` may risk being overwritten and lost. Thus whenever training a new model from scratch make sure that ```model_id``` is unique to create a new checkpoint folder. When continuing training of an already existing model checkpoint, make sure to set the ```model_id``` to the **same** value so that subsequent checkpoint files are saved in the same checkpoint folder (it is not a requirement but makes sense to avoid confusion). 

- To run a model on the full data (either train + val or test or both) simply omit any data limiting arguments, namely ```data_limit, train_data_limit, val_data_limit, test_data_limit```. 

- To specifically switch off either training or test set evaluation one needs to explicitly set the respective argument to ```False```, i.e. either ```model_training=False``` or ```model_evaluation=False```. Defaults for both are ```True```.

- The device is either automatically inferred from the given machine or explicitly handed via argument ```device```. It takes either values ```cpu``` or ```cuda```. To run on multiple GPUs (and if those are available), set argument ```data_parallel=True``` which will activate PyTorch's DataParallel framework. Training etc. should all work smoothly in the same manner as running locally on e.g. a ```cpu```.

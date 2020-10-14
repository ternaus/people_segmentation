# Binary segmentation of people

![](https://habrastorage.org/webt/bc/eg/g8/bcegg8zdgd-co-lip6hxn976jdm.jpeg)

## Installation

`pip install -U people_segmentation`


### Example inference

Jupyter notebook with the example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZHQ3beJP-7Pbq4I5Jsc8Co2dIkK31ALi?usp=sharing)

## Data
### Train set:

* Mapillary Vistas Commercial 1.2 (train)
* COCO (train)
* Pascal VOC (train)
* [Human Matting](https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets/)

### Validation set:
* Mapillary Vistas Commercial 1.2 (val)
* COCO (val)
* Pascal VOC (val)
* Supervisely

To convert datasets to the format:

```
training
    coco
    matting_humans
    pascal_voc
    vistas

validation
    coco
    pascal_voc
    supervisely
    vistas
```
use this set of [scipts](https://github.com/ternaus/iglovikov_helper_functions/tree/master/iglovikov_helper_functions/data_processing/prepare_people_segmentation).

## Training

### Define the config.
Example at [people_segmentation/configs](people_segmentation/configs)

You can enable / disable datasets that are used for training and validation.

### Define the environmental variable `TRAIN_PATH` that points to the folder with train dataset.
Example:
```bash
export TRAIN_PATH=<path to the tranining folder>
```

### Define the environmental variable `VAL_PATH` that points to the folder with validation dataset.
Example:
```bash
export VAL_PATH=<path to the validation folder>
```

### Training
```
python -m people_segmentation.train -c <path to config>
```

You can check the loss and validation curves for the configs from [people_segmentation/configs](people_segmentation/configs)
at [W&B dashboard](https://wandb.ai/ternaus/people_segmentation-people_segmentation)

### Inference

```bash
python -m torch.distributed.launch --nproc_per_node=<num_gpu> people_segmentation/inference.py \
                                   -i <path to images> \
                                   -c <path to config> \
                                   -w <path to weights> \
                                   -o <output-path> \
                                   --fp16
```

## Web App
https://peoplesegmentation.herokuapp.com/

Code for the web app: https://github.com/ternaus/people_segmentation_demo

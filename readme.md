# Image and Video Analysis Project

## Diffusion models for conditioned generation of human faces

### Authors: Teodoro Sullazzo, Maria Pia Zupi
### Academic year: 2022/2023

This repository contains the code used for the implementation of the project for the Image and Video Analysis course, focusing on conditioned generation of human faces.

The task was carried out using the Imagen base model by [Saharia et al.](https://arxiv.org/pdf/2205.11487.pdf), with the implementation provided by [@lucidrains](https://github.com/lucidrains/imagen-pytorch).

The original code has been reorganized into the following files:

- imagen.py: contains the functioning logic of Imagen
- util.py: contains general utility functions
- models.py: contains the implementation of the underlying models for unet or noise scheduling
- unet.py: contains the implementation of the modified version of unet
- trainer.py: contains the implementation of the Imagen trainer
- trainer_util.py: contains specific utility functions for the trainer

In addition, the following files have been introduced:
- user_train.py: allows the user to train the model
- user_generate_images.py: allows the user to generate images from a trained model
- user_test.py: allows the user to test the quality of the generated images
- main_util.py: shared utility functions among the user_*.py files

The last files rely on the configurations entered in the global_config.json file:

- steps: the number of training steps executed by the trainer
- dim: the number of filters in the outputs of the first layer
- cond_dim: the dimension associated with text conditioning
- dim_mults: the channel dimensions inside the model (multiplied by dim)
- image_sizes: the size of the generated images
- timesteps: the number of steps in the scheduler
- cond_drop_prob: the probability of not applying conditioning
- batch_size
- lr: learning rate
- num_resnet_blocks
- dynamic_thresholding: whether Imagen uses dynamic thresholding in sampling
- data_path: the path of the dataset used for training
- model_save_dir: the path where trained models will be saved
- model_name: the name of the trained model, located in model_save_dir
- convert_from_trainer: whether to load the model from a trainer checkpoint
- image_save_dir: the path to save the generated images
- testing_images_dir: the path where the test images are saved (usually the same as image_save_dir)
- deterministic_generate_images: whether to use a deterministic seed in image generation

- The dataset used is available in the data.nosync.zip file. The labels for testing the model are available in test.txt.

To run locally, you need to install the dependencies first using the following command:

        pip install -r requirements.txt
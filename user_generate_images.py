from main_util import read_config
import torch
import os
from main_util import make_generate
import numpy as np
import random

from transformers import logging
logging.set_verbosity_error()


def generate_images(imagen, text_embeddings, labels, config, names, iter, dir):
  embeds = text_embeddings[labels].cuda()
  images = imagen.sample(text_embeds=embeds, batch_size = config["batch_size"], 
                                    return_pil_images = True,cond_scale = 3.)
  
  for j, img in enumerate(images):
    filename = f'./{dir}/{names[j]}_{iter}.jpg'
    img.save(filename)


def read_test_file(path):
    names = []
    labels = []
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines:
            name,label = line.split(';')
            names.append(name)
            labels.append(int(label,2))
    return names,labels


hyperparams = read_config()

if hyperparams["deterministic_generate_images"]:
    seed = 3128974198
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

trainer,text_embeddings = make_generate(hyperparams)
names,labels_test = read_test_file('test.txt')

if not os.path.exists(hyperparams["image_save_dir"]):
  os.mkdir(hyperparams["image_save_dir"])
  
imagen = trainer.imagen
if hyperparams["convert_from_trainer"]:
    trainer.load(os.path.join(hyperparams["model_save_dir"], hyperparams["model_name"]))
else:
   imagen.load_state_dict(torch.load(os.path.join(hyperparams["model_save_dir"],hyperparams["model_name"])))

print("Model loaded")


for iter in ['A','B','C','D','E','F','G','H','I','J']:
  generate_images(imagen,text_embeddings,
                labels_test,hyperparams,names,iter,hyperparams["image_save_dir"])
from main_util import read_config
import torch
from imagen import Imagen
from unet import Unet
import os
from main_util import make



def generate_images(imagen, text_embeddings, labels, config, names, iter, dir):
  embeds = text_embeddings[labels]
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

imagen,text_embeddings = make(hyperparams)
names,labels_test = read_test_file('test.txt')

if not os.path.exists(hyperparams["image_save_dir"]):
  os.mkdir(hyperparams["image_save_dir"])

imagen.load_state_dict(torch.load(hyperparams["model_save_dir"]))
print("Model loaded")


for iter in ['A','B','C','D','E','F','G','H','I','J']:
  generate_images(imagen,text_embeddings,
                labels_test,hyperparams,names,iter,hyperparams["image_save_dir"])
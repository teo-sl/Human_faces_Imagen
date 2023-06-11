from main_util import get_text_embeddings
import torch
from imagen import Imagen
from unet import Unet
import os


hyperparams = {
    "dim": 128,
    "cond_dim": 128,
    "dim_mults": (1, 2, 4),
    "image_sizes": 64,
    "timesteps": 500,
    "cond_drop_prob": 0.1,
    "batch_size": 15,
    'lr': 1e-4,
    'num_resnet_blocks': 3,
    "dynamic_thresholding": True,
    'model_save_dir': 'path/to/model/dir/',
    'image_save_dir': 'path/to/image/dir/',
}

def make(config):
    labels = [bin(i)[3:] for i in range(2**5,2**(5+1))]
    text_embeddings = get_text_embeddings("faces_embeddings.pkl", labels)

    unet = Unet(
      dim = config["dim"], # the "Z" layer dimension, i.e. the number of filters the outputs to the first layer
      cond_dim = config["cond_dim"],
      dim_mults = config["dim_mults"], # the channel dimensions inside the model (multiplied by dim)
      num_resnet_blocks = config["num_resnet_blocks"],
      layer_attns = (False,) + (True,) * (len(config["dim_mults"]) - 1),
      layer_cross_attns = (False,) + (True,) * (len(config["dim_mults"]) - 1)
    )

    imagen = Imagen(
        unets = unet,
        image_sizes = config["image_sizes"],
        timesteps = config["timesteps"],
        cond_drop_prob = config["cond_drop_prob"],
        dynamic_thresholding = config["dynamic_thresholding"],
    ).cuda()


    return imagen,text_embeddings

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

imagen,text_embeddings = make(hyperparams)
names,labels_test = read_test_file('test.txt')

if not os.path.exists(hyperparams["image_save_dir"]):
  os.mkdir(hyperparams["image_save_dir"])

imagen.load_state_dict(torch.load(hyperparams["model_save_dir"]))
print("Model loaded")


for iter in ['A','B','C','D','E','F','G','H','I','J']:
  generate_images(imagen,text_embeddings,
                labels_test,hyperparams,names,iter,hyperparams["image_save_dir"])
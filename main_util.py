import numpy as np
import random
import requests
import torch
from unet import Unet
from trainer import ImagenTrainer
from imagen import Imagen
from torchvision import transforms as T
from transformers import T5Tokenizer, T5EncoderModel
from einops import rearrange
import os
from PIL import Image
from unet import Unet
from trainer import ImagenTrainer
import json


labels = [bin(i)[3:] for i in range(2**5,2**(5+1))]

def read_config(path='./global_config.json'):
    with open(path, 'r') as f:
        config = json.load(f)
    config = {k: v for k, v in config.items()}
    config['dim_mults'] = tuple(config['dim_mults'])
    config['dynamic_thresholding'] = bool(config['dynamic_thresholding'])
    config['convert_from_trainer'] = bool(config['convert_from_trainer'])
    config['deterministic_generate_images'] = bool(config['deterministic_generate_images'])
    return config



def send_to_telegram(image,bot_token,chat_id):
    img = open(image, 'rb')
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto?chat_id={chat_id}'
    print(requests.get(url, files={'photo': img}))


def get_text_embeddings(name, labels, max_length = 256):
    if os.path.isfile(name):
        return torch.load(name)
    
    model_name = 'google/t5-v1_1-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=max_length)

    model = T5EncoderModel.from_pretrained(model_name)
    model.eval()
    
    def photo_prefix(noun):
        
        ret = "a photo of "
        
        if noun[3]=='1':
            ret+="a smiling "
        else:
            ret+="a frowning "
        if noun[4]=='1':
            ret+="young person "
        else:
            ret+="elderly person "
        if noun[0]=='1':
            ret+="with bangs, "
        else:
            ret+="without bangs, "
        if noun[1]=='1':
            ret+="in glasses "
        else:
            ret+="without glasses "
        if noun[2]=='1':
            ret+="and with a beard"
        else:
            ret+="and shaved"

        return ret
            
    texts = [photo_prefix(x) for x in labels]
    
    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = max_length,
        truncation = True
    )
    
    with torch.no_grad():
        output = model(input_ids=encoded.input_ids , attention_mask=encoded.attention_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = encoded.attention_mask.bool()
    
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.)
    
    torch.save(encoded_text, name)
    
    return encoded_text


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, embeddings, transform=None):
        self.embeddings = embeddings
        self.root = root
        self.transform = transform
        info = os.path.join(root,'train.txt')
        self.images_path = []
        self.images_labels = []
        with open(info, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name, label = line.split(';')
                self.images_path.append(os.path.join(root, img_name))
                self.images_labels.append(int(label,2))
        self.images = []
        for img_path in self.images_path:
            img = Image.open(img_path).convert('RGB')
            self.images.append(img)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        text_embedding = self.embeddings[self.images_labels[idx]]
        return img, text_embedding.clone()
    

def make(config):
    text_embeddings = get_text_embeddings("faces_embeddings.pkl", labels)

    unet = Unet(
      dim = config["dim"], # the "Z" layer dimension, i.e. the number of filters the outputs to the first layer
      cond_dim = config["cond_dim"],
      text_embed_dim = 768,
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

    trainer = ImagenTrainer(imagen, lr=config["lr"])

    ds = FaceDataset(config["data_path"], text_embeddings, transform=T.Compose([ T.RandomHorizontalFlip(), T.ToTensor()]))

    trainer.add_train_dataset(ds, batch_size = config["batch_size"])

    return trainer, text_embeddings



def train(trainer, text_embeddings, config, save_every = 5_000, sample_every = 20_000):
    dummy_filename = './test.jpg'
    model_name = "model.ckpt"
    
    for i in range(config["steps"]):
        loss = trainer.train_step(max_batch_size = config["batch_size"])

        print(f'train_loss {loss}, step {i}')
        
        if  i!=0 and  i % sample_every == 0:
            images = trainer.sample(text_embeds=text_embeddings, batch_size = config["batch_size"], return_pil_images = True)
            for img in images:
                img.save(dummy_filename)
                send_to_telegram(dummy_filename, config["bot_token"],config["chat_id"])

        filename = os.path.join(config["model_save_dir"], model_name)

        if save_every is not None and i != 0 and i % save_every == 0:
            if os.path.exists(filename):
                os.remove(filename)
            trainer.save(filename)


def make_generate(config):
    text_embeddings = get_text_embeddings("faces_embeddings.pkl", labels)

    unet = Unet(
      dim = config["dim"], # the "Z" layer dimension, i.e. the number of filters the outputs to the first layer
      cond_dim = config["cond_dim"],
      text_embed_dim = 768,
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

    trainer = ImagenTrainer(imagen, lr=config["lr"])


    return trainer, text_embeddings

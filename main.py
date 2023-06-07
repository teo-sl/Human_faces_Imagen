from main_util import *

from transformers import logging
logging.set_verbosity_error()


seed = 3128974198
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


hyperparams = {
    "steps": 500_000,
    "dim": 128,
    "cond_dim": 128,
    "dim_mults": (1, 2, 4),
    "image_sizes": 64,
    "timesteps": 500,
    "cond_drop_prob": 0.1,
    "batch_size": 15,
    'lr': 1e-4,
    'num_resnet_blocks': 3,
    "model_save_dir": './model_dir/',
    "dynamic_thresholding": True,
    "use_telegram" : False,
    "bot_token": "...",
    "chat_id": "...", 
    "data_path"  : "path/to/data",
}


model_save_dir = "./model_dir/"
if not os.path.exists(model_save_dir):
  os.mkdir(model_save_dir)

trainer, embeddings = make(hyperparams)

train(trainer, embeddings, hyperparams, save_every=10_000)




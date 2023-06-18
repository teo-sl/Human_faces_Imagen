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


hyperparams = read_config()


model_save_dir = hyperparams["model_save_dir"]
if not os.path.exists(model_save_dir):
  os.mkdir(model_save_dir)

trainer, embeddings = make(hyperparams)

train(trainer, embeddings, hyperparams, save_every=10_000)




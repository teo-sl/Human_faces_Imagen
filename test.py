import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from collections import defaultdict
from tqdm import tqdm
from PIL import Image as PILImage

config = {
    'folder_name' : 'path/to/images',
}



CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")
print('DEVICE', device)

class RealClassifierModel(nn.Module):
    def __init__(self, input_dim=3*64*64):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
            

def group_images(fname_iter):
    
    groups = defaultdict(list)
    
    for fname in fname_iter:
        i = fname.name.index('_')
        
        image_name = fname.name[:i]
        
        groups[image_name].append(fname)
        
    
    assert len(groups) == 50, 'missing images'
    assert sum([1 if len(x) != 10 else 0 for x in groups.values()]) == 0, 'missing samples'
    
    return groups


def load_images(image_list):
    
    tr = transforms.ToTensor()
    
    x_list = [tr(PILImage.open(str(x))).unsqueeze(0) for x in image_list]
    
    return torch.vstack(x_list)
    

def main(folder_name):
    model_path = 'model.pth'
    
    test_set_directives = group_images(Path(folder_name).glob('*.jpg'))
    print(f'{len(test_set_directives)=}')
    
    model = RealClassifierModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    all_outputs_prob = []

    with torch.no_grad():

        for image_list in tqdm(test_set_directives.values()):

            inputs = load_images(image_list)
      

            # Generate outputs
            outputs = model(inputs.to(device))
            all_outputs_prob.append(outputs.detach().cpu().max().item())
            
    print('\n\n\n')
    print(f'Classification: {np.mean(all_outputs_prob)}')


output = main(config['folder_name'])
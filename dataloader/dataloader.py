from typing import Dict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class AbstractArt512(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.data_list = sorted(os.listdir(data_root))
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_fp = os.path.join(self.data_root, self.data_list[idx])
        sample = Image.open(data_fp)
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_dataloader(config:Dict):
    cfg_transform = config['transform']
    cfg_loader = config['dataloader']

    transform_list = [transforms.Resize((cfg_transform['input_size'], cfg_transform['input_size']))]
    if cfg_transform['ToTensor']: transform_list.append(transforms.ToTensor())
    if cfg_transform['Normalize']: transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)
    
    if cfg_loader['dataset'] == 'abstract_art_512':
        data_root = f'dataset/{cfg_loader["dataset"]}'
        dataloader = DataLoader(dataset=AbstractArt512(data_root=data_root,
                                                       transform=transform),
                                batch_size=cfg_loader['batch_size'],
                                shuffle=cfg_loader['shuffle'],
                                num_workers=cfg_loader['num_workers'],
                                drop_last=cfg_loader['drop_last'])
    else:
        raise Exception(f'dataset {cfg_loader["dataset"]} not found.')
    return dataloader
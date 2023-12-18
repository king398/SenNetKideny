import numpy as np

from utils import *
import glob
from model import *
from augmentations import get_valid_transform
from dataset import ImageDatasetOOF
from torch.utils.data import DataLoader
from train_fn import oof_fn
import pandas as pd
import argparse
from pathlib import Path


def main(cfg: dict):
    seed_everything(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_dir = sorted(glob.glob(f"{cfg['validation_dir']}/images/*.tif"))
    validation_dir_xz = sorted(glob.glob(f"{cfg['validation_dir']}_xz/images/*.tif"))
    validation_dir_yz = sorted(glob.glob(f"{cfg['validation_dir']}_yz/images/*.tif"))
    validation_images_stacked = np.stack([cv2.imread(i) for i in validation_dir])

    model = return_model(cfg['model_name'], cfg['in_channels'], cfg['classes'])
    model.load_state_dict(torch.load(f"{cfg['model_dir']}/model.pth", map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    test_dataset = ImageDatasetOOF(validation_dir, get_valid_transform(), volume=validation_images_stacked)
    test_dataset_xz = ImageDatasetOOF(validation_dir_xz, get_valid_transform(height=1056, width=1536), mode='xz',
                                      volume=validation_images_stacked)
    test_dataset_yz = ImageDatasetOOF(validation_dir_yz, get_valid_transform(height=1056, width=1728),
                                      volume=validation_images_stacked, mode='yz')
    test_loader = DataLoader(test_dataset, batch_size=int(cfg['batch_size']), shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=True)
    test_loader_xz = DataLoader(test_dataset_xz, batch_size=int(cfg['batch_size']), shuffle=False,
                                num_workers=cfg['num_workers'], pin_memory=True)
    test_loader_yz = DataLoader(test_dataset_yz, batch_size=int(cfg['batch_size']), shuffle=False,
                                num_workers=cfg['num_workers'], pin_memory=True)

    rles_list, image_ids, volume = oof_fn(model=model, data_loader=test_loader, device=device,
                                          volume_shape=validation_images_stacked.shape[:3],
                                          data_loader_xz=test_loader_xz, data_loader_yz=test_loader_yz)
    oof = pd.DataFrame()
    oof['id'] = image_ids
    oof['rle'] = rles_list
    oof.to_csv(f'{cfg["model_dir"]}/oof.csv', index=False)
    np.savez_compressed(f'{cfg["model_dir"]}/volume.npz', volume=volume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)

from utils import *
from torch.nn import Module
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

    model = return_model(cfg['model_name'], cfg['in_channels'], cfg['classes'])
    model.load_state_dict(torch.load(f"{cfg['model_dir']}/model.pth", map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    test_dataset = ImageDatasetOOF(validation_dir, get_valid_transform(cfg['image_size']))
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False,
                             num_workers=cfg['num_workers'], pin_memory=True)
    rles_list, image_ids = oof_fn(model, test_loader, device)
    oof = pd.DataFrame()
    oof['id'] = image_ids
    oof['rle'] = rles_list
    oof.to_csv(f'{cfg["model_dir"]}/oof.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='config.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)

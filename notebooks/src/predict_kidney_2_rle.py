import pandas as pd
from model import ReturnModel
from utils import *
from dataset import ImageDatasetOOF
from augmentations import get_valid_transform
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

kidney_rles = pd.read_csv("/home/mithil/PycharmProjects/SenNetKideny/data/train_rles_kidneys.csv")
kidney_rles = {kidney_rles['id'][i]: kidney_rles['kidney_rle'][i] for i in range(len(kidney_rles))}
model = ReturnModel('tu-timm/maxvit_base_tf_224.in1k', 3, 2, 224)
model.load_state_dict(torch.load(
    '/home/mithil/PycharmProjects/SenNetKideny/models/maxvit_base_tf_224_volume_normalize_dice_kidney_3_dense/model_epoch_3.pth'))
model = model.cuda()
train_images_2, train_masks_2, train_kidneys_rle_2, train_volume_2 = load_images_and_masks(
    "/home/mithil/PycharmProjects/SenNetKideny/data/train/kidney_2", 'images', 'labels', kidney_rles, 'kidney_2'
)
train_dataset_2 = ImageDatasetOOF(train_images_2, get_valid_transform(), train_volume_2, mode="xy")

train_loader_2 = DataLoader(train_dataset_2, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

rles_list_kidneys = []
for i, (images, image_shapes, image_ids) in tqdm(enumerate(train_loader_2), total=len(train_loader_2)):
    images = images['image'].cuda().float()
    print(image_ids)
    with torch.no_grad() and autocast():
        outputs = model(images)
        outputs = torch.sigmoid(outputs)[:, 1, :, :]
        outputs = outputs.detach().cpu().numpy()
        for index in range(len(outputs)):
            outputs[index] = choose_biggest_object(outputs[index], 0.5)

            kidney_rles[image_ids[index]] = rle_encode(outputs[index])

# convert the dictionary to a dataframe
df = pd.DataFrame(columns=['id','kidney_rle'])
df['id'] = kidney_rles.keys()
df['kidney_rle'] = kidney_rles.values()
df.to_csv("/home/mithil/PycharmProjects/SenNetKideny/data/train_rles_kidneys.csv",index=False)

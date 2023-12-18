import sys, os

sys.path.append('/kaggle/input/blood-vessel-segmentation-third-party')
sys.path.append('/kaggle/input/blood-vessel-segmentation-00')

import cv2
import pandas as pd
from glob import glob
import numpy as np

from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

print('IMPORT OK  !!!!')


class dotdict(dict):
    """ Dot notation access to dictionary attributes """

    def __init__(self, **kwargs):
        super(dotdict, self).__init__(kwargs)
        self.__dict__ = self

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


cfg = dotdict(
    batch_size=12,
    p_threshold=0.10,
    cc_threshold=-1,
)

mode = 'local'  # 'local' #

data_dir = "/home/mithil/PycharmProjects/SenNetKideny/data/"


# -----
def file_to_id(f):
    s = f.split('/')
    return s[-3] + '_' + s[-1][:-4]


if 'local' in mode:
    valid_folder = [
        ('kidney_3_sparse', (496, 996 + 1)),
        # ('kidney_1_dense', (0, 1000+1)),
    ]  # debug for local development

    valid_meta = []
    for image_folder, image_no in valid_folder:
        file = [f'{data_dir}/train/{image_folder}/images/{i:04d}.tif' for i in range(*image_no)]
        H, W = cv2.imread(file[0], cv2.IMREAD_GRAYSCALE).shape
        valid_meta.append(dotdict(
            name=image_folder,
            file=file,
            shape=(len(file), H, W),
            id=[file_to_id(f) for f in file],
        ))

if 'submit' in mode:
    valid_meta = []
    valid_folder = sorted(glob(f'{data_dir}/test/*'))
    for image_folder in valid_folder:
        file = sorted(glob(f'{image_folder}/images/*.tif'))
        H, W = cv2.imread(file[0], cv2.IMREAD_GRAYSCALE).shape
        valid_meta.append(dotdict(
            name=image_folder,
            file=file,
            shape=(len(file), H, W),
            id=[file_to_id(f) for f in file],
        ))

#     glob_file = glob(f'{data_dir}/kidney_5/images/*.tif')
#     if len(glob_file)==3:
#         mode = 'submit-fake' #fake submission to save gpu time when submitting
#         #todo .....


print('len(valid_meta) :', len(valid_meta))
print(valid_meta[0].file[:3])

print('MODE OK  !!!!')


class MyLoader(object):
    def __init__(self, meta):
        self.meta = meta
        self.split = np.array_split(meta.file, max(1, int(len(meta.file) // cfg.batch_size)))

    def __len__(self, ):
        return len(self.split)

    def __getitem__(self, index):
        file = self.split[index]

        image = []
        for f in file:
            m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

            # ---
            # process image
            m = (m - m.min()) / (m.max() - m.min() + 0.001)

            # ---
            image.append(m)

        image = np.stack(image)
        image = torch.from_numpy(image).float().unsqueeze(1)
        return image


print('DATASET OK  !!!!')


def make_dummy_submission():
    submission_df = []
    for d in valid_meta:
        submission_df.append(
            pd.DataFrame(data={
                'id': d['id'],
                'rle': ['1 0'] * len(d['id']),
            })
        )
    submission_df = pd.concat(submission_df)
    submission_df.to_csv('submission.csv', index=False)
    print(submission_df)


# https://www.kaggle.com/competitions/blood-vessel-segmentation/discussion/456033
def choose_biggest_object(mask, threshold):
    mask = ((mask > threshold) * 255).astype(np.uint8)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_label = -1
    max_area = -1
    for l in range(1, num_label):
        if stats[l, cv2.CC_STAT_AREA] >= max_area:
            max_area = stats[l, cv2.CC_STAT_AREA]
            max_label = l
    processed = (label == max_label).astype(np.uint8)
    return processed


def remove_small_objects(mask, min_size, threshold):
    mask = ((mask > threshold) * 255).astype(np.uint8)
    # find all connected components (labels)
    num_label, label, stats, centroid = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # create a mask where small objects are removed
    processed = np.zeros_like(mask)
    for l in range(1, num_label):
        if stats[l, cv2.CC_STAT_AREA] >= min_size:
            processed[label == l] = 1
    return processed


def rle_encode(mask):
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle


# -------------------------------
import segmentation_models_pytorch as smp


def return_model(model_name: str, in_channels=3, classes=2):
    model = smp.Unet(
        encoder_name=model_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,

    )
    return model


checkpoint_file = '/home/mithil/PycharmProjects/SenNetKideny/models/seresnext101d_32x8d_pad_kidney_multiview/model.pth'

net = return_model("tu-seresnext101d_32x8d")
# run_check_net()
state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
print(net.load_state_dict(state_dict, strict=False))  # True

net = net.eval()
net = net.cuda()


# net = torch.compile(net)
def pad_to_nearest_32(array):
    """
    Pads the height and width of an array to the nearest multiple of 32.

    Args:
    - array (numpy.ndarray): The input array of shape [Depth, Height, Width, Channels].

    Returns:
    - numpy.ndarray: Padded array.
    - tuple: Original shape of the array.
    """
    B, C, H, W, = array.shape

    # Calculate new dimensions as the nearest multiple of 32
    new_H = ((H - 1) // 32 + 1) * 32
    new_W = ((W - 1) // 32 + 1) * 32

    # Calculate padding for H and W
    pad_H = (new_H - H) // 2
    pad_W = (new_W - W) // 2

    # Apply padding
    padded_array = np.pad(array, ((0, 0), (pad_H, pad_H), (pad_W, pad_W), (0, 0)), mode='constant', constant_values=0)

    return padded_array, (B, H, W, C)


def pad_to_nearest_32_next(array):
    """
    Pads the height and width of an array to the nearest multiple of 32 for an array with shape [Depth, Channels, Height, Width].
    Args:
    - array (numpy.ndarray): The input array of shape [Depth, Channels, Height, Width].
    Returns:
    - numpy.ndarray: Padded array with the height and width dimensions padded to the nearest multiple of 32.
    - tuple: Original shape of the array.
    """
    D, C, H, W = array.shape
    # Calculate new dimensions as the nearest multiple of 32
    new_H = ((H + 31) // 32) * 32
    new_W = ((W + 31) // 32) * 32
    # Calculate padding for H and W
    pad_H = new_H - H
    pad_W = new_W - W
    # Apply padding only to the H and W dimensions
    padded_array = np.pad(array, ((0, 0), (0, 0), (0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
    return padded_array, (D, C, H, W)


def remove_padding(padded_array, original_shape):
    """
    Removes padding from an array to return it to its original shape.

    Args:
    - padded_array (numpy.ndarray): The padded array.
    - original_shape (tuple): The original shape of the array (Depth, Height, Width).

    Returns:
    - numpy.ndarray: Array with padding removed.
    """
    D, H, W, C = original_shape
    new_D, new_H, new_W = padded_array.shape

    # Calculate starting indices
    start_H = (new_H - H) // 2
    start_W = (new_W - W) // 2

    # Remove padding
    unpadded_array = padded_array[:, start_H:start_H + H, start_W:start_W + W]

    return unpadded_array


def remove_padding_tensor(padded_tensor, original_shape, axis):
    """
    Removes padding from a tensor to return it to its original shape.

    Args:
    - padded_tensor (torch.Tensor): The padded tensor.
    - original_shape (tuple): The original shape of the tensor (Depth, Height, Width).

    Returns:
    - torch.Tensor: Tensor with padding removed.
    """
    print(padded_tensor.shape)
    print(original_shape)
    H, W = original_shape

    Batch_size, C, new_H, new_W = padded_tensor.shape

    # Calculate starting indices
    start_H = (new_H - H) // 2
    start_W = (new_W - W) // 2

    # Remove padding
    unpadded_tensor = padded_tensor[:, :, start_H:start_H + H, start_W:start_W + W]

    return unpadded_tensor


def do_submit():
    submission_df = []
    for d in valid_meta:
        volume = [cv2.cvtColor(cv2.imread(f, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in d.file]
        volume = np.stack(volume)
        D, H, W, C = volume.shape
        volume_x, shape = pad_to_nearest_32(volume)
        predict = np.zeros((D, H, W), dtype=np.float16)
        axes = [0, 1, 2]  # [2]  #
        for axis in axes:  # 0
            loader = np.array_split(np.arange((D, H, W)[axis]), max(1, int((D, H, W)[axis] // cfg.batch_size)))
            num_valid = len(loader)

            B = 0
            start_timer = timer()
            for t in range(num_valid):

                if axis == 0:
                    image = volume[loader[t].tolist()]
                    image = image.transpose(0, 3, 1, 2)
                if axis == 1:
                    image = volume[:, loader[t].tolist()]
                    image = image.transpose(1, 3, 0, 2)
                if axis == 2:
                    image = volume[:, :, loader[t].tolist()]
                    print(image.shape)

                    image = image.transpose(2, 3, 0, 1, )
                    # pad the 2nd dimension to nearest factor of 32 only the second dimension
                print(image.shape)
                image, shape_new = pad_to_nearest_32_next(image)
                print(image.shape)
                batch_size, bc, bh, bw, = image.shape
                m = image.reshape(batch_size, -1)
                m = (m - m.min(keepdims=True)) / (m.max(keepdims=True) - m.min(keepdims=True) + 0.001)
                m = m.reshape(batch_size, bc, bh, bw, )
                m = np.ascontiguousarray(m)
                image = torch.from_numpy(m).float().cuda()
                print(image.shape)
                print()
                # ----
                counter = 0
                vessel, kidney = 0, 0
                image = image.cuda()
                with torch.cuda.amp.autocast(enabled=True):
                    with torch.no_grad():
                        v, k = remove_padding_tensor(net(image).sigmoid(), (shape_new[2], shape_new[3]), axis=axis).split(1,
                                                                                                                dim=1)
                        print(v.max(),k.max())
                        vessel += v
                        kidney += k
                        counter += 1

                        v, k = remove_padding_tensor(net(torch.flip(image, dims=[2, ])).sigmoid(), (shape_new[2], shape_new[3]),
                                                     axis=axis).split(1,
                                                                      dim=1)
                        vessel += torch.flip(v, dims=[2, ])
                        kidney += torch.flip(k, dims=[2, ])
                        counter += 1

                        v, k = remove_padding_tensor(net(torch.flip(image, dims=[3, ])).sigmoid(), (shape_new[2], shape_new[3]),
                                                     axis=axis).split(1,
                                                                      dim=1)
                        vessel += torch.flip(v, dims=[3, ])
                        kidney += torch.flip(k, dims=[3, ])
                        counter += 1

                        v, k = remove_padding_tensor(net(torch.rot90(image, k=1, dims=[2, 3])).sigmoid(),
                                                     (shape_new[3], shape_new[2]),
                                                     axis=axis).split(1,
                                                                      dim=1)
                        vessel += torch.rot90(v, k=-1, dims=[2, 3])
                        kidney += torch.rot90(k, k=-1, dims=[2, 3])
                        counter += 1

                        v, k = remove_padding_tensor(net(torch.rot90(image, k=2, dims=[2, 3])).sigmoid(),
                                                     (shape_new[2], shape_new[3]),
                                                     axis=axis).split(1,
                                                                      dim=1)
                        vessel += torch.rot90(v, k=-2, dims=[2, 3])
                        kidney += torch.rot90(k, k=-2, dims=[2, 3])
                        counter += 1

                        v, k = remove_padding_tensor(net(torch.rot90(image, k=3, dims=[2, 3])).sigmoid(),
                                                     (shape_new[3], shape_new[2]),
                                                     axis=axis).split(1,
                                                                      dim=1)
                        vessel += torch.rot90(v, k=-3, dims=[2, 3])
                        kidney += torch.rot90(k, k=-3, dims=[2, 3])
                        counter += 1
                print(vessel)

                vessel = vessel / counter
                kidney = kidney / counter
                # print(i, image.shape, mask.shape)

                vessel = vessel.float().data.cpu().numpy()
                kidney = kidney.float().data.cpu().numpy()

                # ----------------------------------------
                batch_size = len(vessel)
                for b in range(batch_size):
                    mk = kidney[b, 0]
                    mk = choose_biggest_object(mk, threshold=0.5)
                    mv = vessel[b, 0]
                    p = (mv * mk)
                    if axis == 0:
                        predict[B + b] += p
                    if axis == 1:
                        predict[:, B + b] += p
                    if axis == 2:
                        predict[:, :, B + b] += p

                    # debug only

                    # plt.waitforbuttonpress()

                # ----------------------------------------
                B += batch_size

        print('')
        predict = remove_padding(predict, shape)
        predict = predict / 2

        predict = (predict > 0.10).astype(np.uint8)

        # post processing ---

        rle = [rle_encode(p) for p in predict]
        submission_df.append(
            pd.DataFrame(data={
                'id': d['id'],
                'rle': rle,
            })
        )

    submission_df = pd.concat(submission_df)
    submission_df.to_csv('submission.csv', index=False)
    print(submission_df)


glob_file = glob(f'{data_dir}/test/kidney_5/images/*.tif')
if (mode == 'submit') and (len(glob_file) == 3):  # cannot do 3d cnn because too few test files
    make_dummy_submission()
else:
    do_submit()

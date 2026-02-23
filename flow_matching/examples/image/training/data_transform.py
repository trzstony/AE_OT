# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    RandomHorizontalFlip,
    Resize,
    ToDtype,
    ToImage,
)


def get_train_transform(
    image_size: int = 64,
    center_crop_size: int = 178,
    random_hflip: bool = False,
):
    transform_list = [
        ToImage(),
        CenterCrop(center_crop_size),
        Resize((image_size, image_size)),
        ToDtype(torch.float32, scale=True),
    ]
    if random_hflip:
        transform_list.insert(1, RandomHorizontalFlip())
    return Compose(transform_list)

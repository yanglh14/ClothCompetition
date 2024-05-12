# script to segment the cloth using "segment_anything"
#
# Zhang Zeqing
# 2025/05/09

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from segment_anything import SamPredictor, sam_model_registry


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

import os
current_dir = os.path.dirname(__file__)
cloth_comp_dir = os.path.dirname(os.path.dirname(current_dir))
path_to_checkpoint = os.path.join(cloth_comp_dir, "pth", "sam_vit_h_4b8939.pth") # "/home/alan/BionicDL/ClothCompetition/ClothCompetition/pth/sam_vit_h_4b8939.pth"
model_type = "vit_h"
dir_image = os.path.join(cloth_comp_dir, "real_robot", "log", "mask_comp_failure.png") # "/home/yang/Projects/ClothCompetition/ClothCompetition/real_robot/log/mask_comp_failure.png"

## load image
image = cv2.imread(dir_image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
## show example image
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=path_to_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

## find a specific object with ONE point
# input_point = np.array([[345, 226]])
# input_label = np.array([1])
input_point = np.array([[345, 226], [345, 62]])
input_label = np.array([1, 1])
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)
# print("masks.shape: ", masks.shape)  # (number_of_masks) x H x W
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks, plt.gca())
# show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()


## find a specific object with a BOX
# input_box = np.array([265, 480, 415, 0])
# masks, _, _ = predictor.predict(
#     point_coords=None,
#     point_labels=None,
#     box=input_box[None, :],
#     multimask_output=False,
# )
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_mask(masks[0], plt.gca())
# show_box(input_box, plt.gca())
# plt.axis('off')
# plt.show()

## find a specific object with a BOX and a point
# input_box = np.array([265, 480, 415, 0])
# input_point = np.array([[345, 226]])
# input_label = np.array([1])
# masks, _, _ = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     box=input_box,
#     multimask_output=False,
# )
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_mask(masks[0], plt.gca())
# show_box(input_box, plt.gca())
# show_points(input_point, input_label, plt.gca())
# plt.axis('off')
# plt.show()
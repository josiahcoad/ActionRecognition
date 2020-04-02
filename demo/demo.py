# pylint: disable=no-member
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
from torchvision import transforms, models
import torchvision

import os
import json
import argparse

from utils import *

# Load Model
# Load face detector
mtcnn = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

# 'My Drive/CCSE 689/images/myimg.png' -> myimg
def get_basename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


# Give probability of shouting for every image in image folder
def predict(video):
    # [D, W, H, C] -> [D, C, W, H]
    frames = torch.transpose(torch.transpose(video, 1, 2), 1, 3)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((144, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transformed = torch.stack([transform(frame) for frame in frames])
    out = model(transformed)
    return torch.nn.functional.softmax(out.data, dim=1).numpy().T[1]


# make time-probability graph
def mk_prob_graph(predictions, title, outfolder):
    # predictions: [[ts, prob], ...]
    plt.plot(*predictions.T, marker='.')
    plt.title(title)
    plt.xlabel('seconds')
    plt.ylim(0, 1)
    plt.ylabel('shouting probability')
    plt.savefig(os.path.join(outfolder, title + '.png'))


# make json file
def mk_json(predictions, title, outfolder):
    data = {'shouting': [[ts, float(prob)] for ts, prob in predictions]}
    with open(os.path.join(outfolder, title + '.json'), 'w') as f:
        f.write(json.dumps(data))

def labelvid(vidpath, resultspath):
    facelist, frames, timestamps = detection_pipeline(vidpath)
    print("Video processed")
    probs = [0 if faces is None else get_frame_prob(faces) for faces in facelist]
    print("Predictions done")
    title = get_basename(vidpath)
    show_frames(frames, probs, outfolder=resultspath, title=title)
    show_timestamps(probs, timestamps, outfolder=resultspath, title=title)
    mk_json(zip(timestamps, probs), title, resultspath)

# get vidpath, resultspath from command line if provided, else default
def parseargs():
    parser = argparse.ArgumentParser(
        description='Demo for shouting action recognition in videos.')
    parser.add_argument('--vidpath', help='The path to an .mp4 file.')
    parser.add_argument(
        '--resultspath', help='The path to a folder to store the results of analysis.')
    args = parser.parse_args()
    vidpath = args.vidpath or './videos/demo.mov'
    resultspath = args.resultspath or './results'
    return vidpath, resultspath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo for shouting action recognition in videos.')
    parser.add_argument('--vidpath', help='The path to an .mp4 file.')
    parser.add_argument(
        '--resultspath', help='The path to a folder to store the results of analysis.')
    labelvid(*parseargs())

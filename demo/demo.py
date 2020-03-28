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

# Load Model
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

checkpoint = torch.load('./assets/model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])


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


# make gif with predictions
def mk_gif(predictions, frames, spf, title, outfolder):
    fig = plt.figure(figsize=(13, 10))
    ims = []
    for img, (_, prob) in zip(frames, predictions):
        im = plt.imshow(img, animated=True)
        plt.axis('off')
        t = plt.text(100, 100, round(prob, 2), color='red', fontsize=20)
        ims.append([im, t])
    anim = animation.ArtistAnimation(fig, ims, interval=spf*1000, blit=True)
    anim.save(os.path.join(outfolder, title + '.mp4'))
    plt.close()


def downsample(vid, orig_fps, dest_fps):
    assert orig_fps >= dest_fps
    skip = orig_fps // dest_fps
    return torch.stack([frame for i, frame in enumerate(vid) if i % skip == 0])


# get prediction results for video
def labelvid(vidpath, resultspath):
    # Use model to predict for each image in seqence
    assert os.path.exists(vidpath), f'{vidpath} does not exist'
    video, _, data = torchvision.io.read_video(vidpath, pts_unit='sec')
    orig_fps = data['video_fps']
    dest_fps = 3
    spf = 1 / dest_fps
    print(f'video read ({len(video)} frames)')
    video = downsample(video, orig_fps, dest_fps)
    print(f'video downsampled ({len(video)} frames)')
    probs = predict(video)
    predictions = np.array([(i * spf, p) for i, p in enumerate(probs)])
    print('predictions finished')
    # Create results
    vidname = get_basename(vidpath)
    mk_prob_graph(predictions, vidname, resultspath)
    mk_json(predictions, vidname, resultspath)
    mk_gif(predictions, video, spf, vidname, resultspath)


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

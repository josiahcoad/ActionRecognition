# @title Import Libs
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

from collections import defaultdict
import matplotlib.image as mpimg

import cv2
import os
import json
from PIL import Image, ImageDraw, ImageFont
import shutil
import argparse


# @title Define Model Class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16,
                              kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(64416, 2)

    def forward(self, x):
        # pdb.set_trace()
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out


# @title Load Model
model = CNNModel()
checkpoint = torch.load('./assets/model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

# @title Custom Dataset Class


class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.files = [x for x in os.listdir(root_dir) if x.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgpath = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(imgpath)
        if self.transform:
            image = self.transform(image)
        return image

    def samples(self, idx):
        return os.path.join(self.root_dir, self.files[idx])


# @title Define Functions
# 'My Drive/CCSE 689/images/myimg.png' -> myimg
def get_basename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


# 'My Drive/CCSE 689/images/this_is_file_11.11211.png' -> (this_is_file, 11.11211)
def parse_imgpath(path):
    base = get_basename(path)
    ts = float(base.split('_')[-1])
    vidname = '_'.join(base.split('_')[:-1])
    return vidname, ts


# Give probability of shouting for every image in image folder
def predict(imgfolder):
    # expects images to be labeled <vidname>_<ts>.ext
    transform = transforms.Compose([transforms.Resize((144, 256)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485], std=[0.229]),
                                    ])
    dataset = InferenceDataset(imgfolder, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    predictions = defaultdict(list)
    for i, inputs in enumerate(dataloader):
        outputs = model(inputs)
        prob = torch.nn.functional.softmax(outputs.data, dim=1).cpu().numpy()[
            0][1]  # image 0, class 1
        path = dataloader.dataset.samples(i)
        vidname, ts = parse_imgpath(path)
        predictions[vidname].append((ts, prob, path))
    # sort predictions
    predictions = {vidname:
                   sorted(vals, key=lambda x: x[0]) for vidname, vals in predictions.items()}
    return predictions


# Show images annotated with probability labels
def show_imgs(results):
    sresults = sorted(results, key=lambda x: x[1][-7:-4])

    plt.figure(figsize=(15, 15))
    for i, (prob, img) in enumerate(sresults, 1):
        ax = plt.subplot(10, 15, i)
        img = mpimg.imread(img)
        plt.imshow(img)
        plt.title(round(prob, 2))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# Convert video to image sequence
def video_to_imgs(vidpath, imgpath):
    # imgpath: folder where images should be put
    vidname = get_basename(vidpath)
    # Make path if it doesn't exist
    if not os.path.exists(imgpath):
        try:
            os.mkdir(imgpath)
        except:
            raise Exception("Not able to make image folder.")
    # play video from file
    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps > 0, 'File does not exist or is not a video format'
    spf = 1.0 / fps  # seconds per frame
    currentFrame = 0
    last_ts = -1
    curr_ts = 0
    while(True):
        curr_ts = currentFrame * spf
        # capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        if curr_ts - last_ts > .3:  # only sample  ~3 fps
            last_ts = curr_ts
            # save image of the current frame in jpg file
            name = '{}_{:5f}.jpg'.format(os.path.join(
                imgpath, vidname), currentFrame * spf)
            cv2.imwrite(name, frame)

        currentFrame += 1

    # release the capture
    cap.release()
    cv2.destroyAllWindows()


# make time-probability graph
def mk_prob_graph(predictions, title, outfolder):
    # predictions: [(ts, prob)]
    timestamps = [x[0] for x in predictions]
    probs = [x[1] for x in predictions]
    plt.plot(timestamps, probs, marker='.')
    plt.title(title)
    plt.xlabel('seconds')
    plt.ylim(0, 1)
    plt.ylabel('shouting probability')
    plt.savefig(os.path.join(outfolder, title + '.png'))


def mk_prob_graphs(predictions, outfolder):
    for vidname, preds in predictions.items():
        mk_prob_graph(preds, vidname, outfolder)


# make json file
def mk_json(predictions, title, outfolder):
    data = {'shouting': [[ts, float(prob)] for ts, prob, _ in predictions]}
    with open(os.path.join(outfolder, title + '.json'), 'w') as f:
        f.write(json.dumps(data))


def mk_jsons(predictions, outfolder):
    for vidname, preds in predictions.items():
        mk_json(preds, vidname, outfolder)


# @title Make gifs
def annotate_img(imgpath, prob):
    # get an image
    base = Image.open(imgpath).convert('RGBA')

    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255, 255, 255, 0))

    # get a font
    fnt = ImageFont.truetype(
        './assets/Roboto-Regular.ttf', 40)

    # get a drawing context
    d = ImageDraw.Draw(txt)

    # draw text, full opacity
    d.text((10, 60), '{:.2f}'.format(
        prob), font=fnt, fill=(255, 0, 0, 255))

    return Image.alpha_composite(base, txt)


def mk_gif(predictions, title, outfolder):
    images = [annotate_img(imgpath, prob) for _, prob, imgpath in predictions]
    duration = (predictions[1][0] - predictions[0][0]) * 1000
    images[0].save(os.path.join(outfolder, title + '.gif'),
                   save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)


def mk_gifs(predictions, outfolder):
    # predictions: {(path, vidname): [(ts, prob)]}
    for vidname, preds in predictions.items():
        mk_gif(preds, vidname, outfolder)


def labelvid(vidpath, imgpath, resultspath):
    """
    (vidpath: Path to the video file to analyze
    imgpath: Path to the folder to put the pictures from the video
    resultspath: Path to the folder to put the results of the predictions
    """
    # Convert video to images
    video_to_imgs(vidpath, imgpath)
    # Use model to predict for each image in seqence
    print(f"Img creation done")
    predictions = predict(imgpath)
    print(f"Prediction done")
    # Create graph and json file
    mk_prob_graphs(predictions, resultspath)
    mk_jsons(predictions, resultspath)
    mk_gifs(predictions, resultspath)
    print(f"Results finished")
    # remove temp img folder
    shutil.rmtree(imgpath)


# get vidpath, imgpath, resultspath from command line if provided, else default
def parseargs():
    parser = argparse.ArgumentParser(
        description='Demo for shouting action recognition in videos.')
    parser.add_argument('--vidpath', help='The path to an .mp4 file.')
    parser.add_argument(
        '--imgpath', help='The path to a folder to store temp data. Parent directory should exist.')
    parser.add_argument(
        '--resultspath', help='The path to a folder to store the results of analysis.')
    args = parser.parse_args()
    vidpath = args.vidpath or './videos/demo.mov'
    imgpath = args.imgpath or './tmp'
    resultspath = args.resultspath or './results'
    return vidpath, imgpath, resultspath


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo for shouting action recognition in videos.')
    parser.add_argument('--vidpath', help='The path to an .mp4 file.')
    parser.add_argument(
        '--imgpath', help='The path to a folder to store temp data. Parent directory should exist.')
    parser.add_argument(
        '--resultspath', help='The path to a folder to store the results of analysis.')
    args = parser.parse_args()
    labelvid(*parseargs())

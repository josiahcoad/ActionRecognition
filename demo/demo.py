from utils import *
from tensorflow import keras
import numpy as np
import argparse
import os

def main(vidpath, resultspath):
    model = keras.models.load_model('assets/model.h5')
    frames, points, ppoints, ts = load_vid(vidpath, 10)
    probs = get_vid_probs(model, ppoints)
    title = get_basename(vidpath)
    tssavepath = os.path.join(resultspath, title + '_ts.png')
    fsavepath = os.path.join(resultspath, title + '_frames.png')
    tsplot(ts, level_vid_probs(probs, 3, 1), tssavepath)
    show_frames(probs, points, frames, fsavepath)



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
    vidpath, resultspath = parseargs()
    assert os.path.exists(vidpath), 'Video path does not exist'
    assert os.path.exists(resultspath), 'Results path does not exist'
    main(vidpath, resultspath)

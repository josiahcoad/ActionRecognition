from utils import *
from tensorflow import keras
import numpy as np
import argparse
import os

from joblib import dump, load

def main(vidpath, resultspath):
    model = keras.models.load_model('assets/model.h5')
    # model = load('assets/svm_personal_acc80.joblib')
    frames, points, ppoints, ts = load_vid(vidpath, 5)
    probs = get_vid_probs(model, ppoints)
    leveled_probs = level_vid_probs(probs, 3, 1)
    title = get_basename(vidpath)
    savepath = os.path.join(resultspath, title)
    tsplot(ts, leveled_probs, savepath + '.png')
    show_frames(probs, points, frames, savepath + '_frames.png')
    tsjson(ts, leveled_probs, savepath + '.json')



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

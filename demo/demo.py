from plotutils import *
from vidutils import load_video_frames, get_face_probs
from audioutils import get_audio_probs
from tensorflow import keras
import joblib
import numpy as np
import argparse
import os
import warnings


def get_current_aud_prob(ts, aud_probs):
    for key in aud_probs:
        if ts >= key:
            return aud_probs[key]


def get_vid_probs(aud_probs, face_probs, timestamps, theta=.5):
    vid_probs = []
    for ts, face_prob in zip(timestamps, face_probs):
        aud_prob = get_current_aud_prob(ts, aud_probs)
        if face_prob is not None:
            vid_probs.append(theta * face_prob + (1-theta) * aud_prob)
        else:
            vid_probs.append(aud_prob)
    return vid_probs


def main(vidpath, resultspath, resultname):
    vmodel = keras.models.load_model('assets/keras_vgg19_84acc.h5')
    amodel = joblib.load('assets/audio_mlp_classifier.joblib')
    frames, faces, timestamps = load_video_frames(vidpath, skip=10)
    face_probs = get_face_probs(vmodel, faces)
    aud_probs = get_audio_probs(amodel, vidpath)
    print('aud_probs', aud_probs)
    print('face_probs', face_probs)
    probs = get_vid_probs(aud_probs, face_probs, timestamps)  # [.2, .3, ...]
    leveled_probs = smooth_probs(probs, 3, 1)
    title = get_basename(vidpath)
    savepath = os.path.join(resultspath, resultname or title)
    tsplot(timestamps, probs, savepath + '.jpg')
    show_frames(probs, frames, savepath + '_frames.png')
    tsjson(timestamps, leveled_probs, savepath + '.json')


# get vidpath, resultspath from command line if provided, else default
def parseargs():
    parser = argparse.ArgumentParser(
        description='Demo for shouting action recognition in videos.')
    parser.add_argument('--vidpath', help='The path to an .mp4 file.')
    parser.add_argument(
        '--resultspath', help='The path to a folder to store the results of analysis.')
    parser.add_argument(
        '--resultname', help='The base name of the result files to be generated. If none, use vidname.')
    args = parser.parse_args()
    vidpath = args.vidpath or './videos/demo.mov'
    resultspath = args.resultspath or './results'
    resultname = args.resultname or None
    return vidpath, resultspath, resultname
    


if __name__ == '__main__':
    vidpath, resultspath, resultname = parseargs()
    assert os.path.exists(vidpath), 'Video path does not exist'
    assert os.path.exists(resultspath), 'Results path does not exist'
    main(vidpath, resultspath, resultname)

from mlxtend.image import extract_face_landmarks
import cv2
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def get_basename(path):
    # 'My Drive/CCSE 689/images/myimg.png' -> myimg
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def vid2frames(path):
    # returns: shape(F,W,H,C)
    cap = cv2.VideoCapture(path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames)


def get_fps(path):
    # returns: shape(F,68,2)
    return cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)


def flatten_faces(faces):
    # (F, 62, 2) -> (F, 136)
    return faces.reshape(-1, 136)


def isface(face):
    # face: any shape
    return not (np.isnan(face).any() or face.sum() == 0)


def scale01(x):
    # x: shape(F, 136) or shape(F, 68, 2)
    return ((x - np.min(x, 1, keepdims=True)) /
            (np.max(x, 1, keepdims=True) - np.min(x, 1, keepdims=True)))


def frames2points(frames):
    return np.array([extract_face_landmarks(frame) for frame in frames])


def load_folder(path, skip=1):
    files = glob.glob(os.path.join(path, '*'))
    vids = []
    for f in files:
        print(f)
        frames = vid2frames(f)
        vids.append(frames2points(frames[::skip]))
    return np.concatenate(vids)


def preprocess(points):
    # frames: shape(F, 68, 2)
    points = scale01(points)
    return flatten_faces(points)


def tsplot(ts, probs, savepath):
    plt.plot(ts, probs)
    plt.hlines(np.nanmean(probs), *plt.xlim(), color='red', alpha=.3)
    plt.annotate(round(np.nanmean(probs), 2), (0, np.nanmean(probs)))
    plt.ylim(-.1, 1.1)
    plt.savefig(savepath)


def tsjson(ts, probs, savepath):
    with open(savepath, 'w') as f:
        f.write(json.dumps({'shouting': np.vstack([ts, probs]).tolist()}))


def singlepred(model, x):
    return model.predict(x.reshape(1, -1))[0][1]


def level_vid_probs(probs, nffill=2, nroll=3):
    # fillna foward {nffill}, then fill remaining na with 0, then get {nroll} rolling avg
    return pd.Series(probs).fillna(method='ffill', limit=nffill).fillna(0).ewm(span=nroll).mean()


def get_vid_probs(model, points):
    return [singlepred(model, face) if isface(face) else np.NaN for face in points]


def load_vid(path, skip=1):
    frames = vid2frames(path)[::skip]
    points = frames2points(frames)
    ppoints = preprocess(points)
    fps = get_fps(path)
    ts = np.arange(len(points)) * skip / fps
    return frames, points, ppoints, ts


def show_frames(probs, points, frames, savepath, nskip=1):
    # probs: shape(F)
    # points: shape(F, 68, 2)
    # frames: shape(F, W, H, 3)
    numimgs = len(frames)
    ncols = 10
    nrows = np.ceil(numimgs / (ncols * nskip))
    plt.figure(figsize=(ncols*2, nrows*2))

    ls = np.array(list(zip(probs, points, frames)))[::nskip]
    for i, (prob, points, frame) in enumerate(ls):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(frame)
        if isface(points):
            plt.scatter(*points.T, s=2)
            plt.title(round(prob, 2))
            # plt.annotate(round(prob, 2), (100, 100), c='red', fontsize=15)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath)

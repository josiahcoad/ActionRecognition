import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import json

def get_basename(path):
    # 'My Drive/CCSE 689/images/myimg.png' -> myimg
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def tsplot(ts, probs, savepath):
    plt.plot(ts, probs)
    plt.hlines(np.nanmean(probs), *plt.xlim(), color='red', alpha=.3)
    plt.annotate(round(np.nanmean(probs), 2), (0, np.nanmean(probs)))
    plt.ylim(-.1, 1.1)
    plt.savefig(savepath)


def tsjson(ts, probs, savepath):
    data = np.vstack([ts, probs]).T.astype(float).round(3).tolist()
    with open(savepath, 'w') as f:
        f.write(json.dumps({'shouting': data}))


def level_vid_probs(probs, nffill=2, nroll=3):
    # fillna foward {nffill}, then fill remaining na with 0, then get {nroll} rolling avg
    return pd.Series(probs).fillna(method='ffill', limit=nffill).fillna(0).ewm(span=nroll).mean()


def show_frames(probs, frames, savepath, nskip=1):
    # probs: shape(F)
    # frames: shape(F, W, H, 3)
    numimgs = len(frames)
    ncols = 10
    nrows = np.ceil(numimgs / (ncols * nskip))
    plt.figure(figsize=(ncols*2, nrows*2))

    ls = list(zip(probs, frames))[::nskip]
    for i, (prob, frame) in enumerate(ls):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(np.array(frame))
        if prob:
            plt.title(round(prob, 2))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath)

from mlxtend.image import extract_face_landmarks
import cv2
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt


# 'My Drive/CCSE 689/images/myimg.png' -> myimg
def get_basename(path):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def get_fps(path):
    return cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)


def vid2frames(path):
    # returns: [(W,H,C), ...]
    cap = cv2.VideoCapture(path)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)
    cap.release()
    cv2.destroyAllWindows()
    return frames


def show_frames(frames, probs, nskip=1, outfolder=None, title=None):
    print(
        f'Probability video is shouting: {round(np.mean([p for p in probs if p]), 2)}')
    numimgs = len(frames)
    ncols = 10
    nrows = np.ceil(numimgs/(ncols * nskip))
    plt.figure(figsize=(ncols, nrows))
    for i, (frame, prob) in enumerate(zip(frames[0:numimgs:nskip], probs[0:numimgs:nskip])):
        plt.subplot(nrows, ncols, i+1)
        plt.title(f'{round(prob, 2)}')
        plt.imshow(frame)
        plt.axis('off')
    plt.tight_layout()
    path = os.path.join(outfolder, title + '_frames.png')
    print('Frames being saved at ' + path)
    plt.savefig(path)
    plt.close()


def intermix(lst1, lst2):
    return np.array([[i, j] for i, j in zip(lst1, lst2)]).ravel()


def scale01(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_scaled_points(face):
    x, y, c = np.array(np.split(face, 70)).T
    if np.mean(c) > 0.2 and (min(x) == 0 or max(x) == 0 or min(y) == 0 or max(y) == 0):
        show_face(intermix(x, y))
    return intermix(scale01(x), scale01(y)) if np.mean(c) > 0.2 else None


def get_frame_probs(path, clf):
    files = glob.glob(path)
    faceframes = []
    for i, name in enumerate(files):
        with open(name) as file:
            js = json.loads(file.read())
            if len(js['people']) > 0:
                faceframes.append(js['people'][0]['face_keypoints_2d'])
            else:
                faceframes.append(None)
    faceframes = [None if frame is None else get_scaled_points(
        np.array(frame)) for frame in faceframes]

    def pred(x): return clf.predict(x.reshape(1, -1)).T[1][0]
    probs = [np.NaN if frame is None else pred(frame) for frame in faceframes]
    return probs


### --------- OUTPUTS ---------- ###

def mk_pred_plot(predictions, outfolder, title):
    plt.plot(*predictions.T)
    plt.ylim(-0.1, 1.1)
    path = os.path.join(outfolder, title + '.png')
    print('Timestamps being saved at ' + path)
    plt.savefig(path)
    plt.close()


def mk_pred_json(predictions, outfolder, title):
    with open(os.path.join(outfolder, title + '.json'), 'w') as f:
        f.write(json.dumps({'shouting': predictions}))

import glob
import time
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import pickle
from joblib import dump, load
from sklearn.decomposition import PCA

# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

# Load face detector
mtcnn = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""
    
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        """Constructor for DetectionPipeline class.

        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def __call__(self, filename):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            filename {str} -- Path to video.
        """
        assert os.path.exists(filename)
        # Create video reader and find length and fps
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = v_cap.get(cv2.CAP_PROP_FPS)

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        batch = []
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                batch.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(batch) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(batch))
                    batch = []
                frames.append(frame)

        v_cap.release()

        return faces, frames, sample * (1 / fps)


# Define face detection pipeline
detection_pipeline = DetectionPipeline(detector=mtcnn, batch_size=60, resize=0.25)

def load_model(load_file=None, X=None, y=None):
  return pickle.loads(load(f'{load_file}.joblib'))


clf = load_model('assets/model_svm_personal')
pca = load_model('assets/model_pca_personal')

def process_faces(faces, resnet):
    # Filter out frames without faces
    faces = [f for f in faces if f is not None]
    if len(faces) == 0:
      return torch.Tensor([])
    faces = torch.cat(faces).to(device)

    # Generate facial feature vectors using a pretrained model
    embeddings = resnet(faces)

    # Calculate centroid for video and distance of each face's feature vector from centroid
    centroid = embeddings.mean(dim=0)
    x = (embeddings - centroid).norm(dim=1).cpu()
    
    return embeddings.cpu()


def scale(x):
  return (x - x.min()) / (x.max() - x.min())


#@title show_face_preds
def predict_face_probs(facelist, resnet, pca, clf):
  # get 512 features from resnet
  with torch.no_grad():
    features = process_faces(facelist, resnet)
  # reduce to 20 features using PCA
  embeddings = pca.transform(features)
  # use svm on 20 features to predict positive probability
  return clf.predict_proba(embeddings).T[1]


def show_timestamps(probs, timestamps, outfolder, title):
  probs = pd.Series(probs).ewm(span=10).mean().values
  plt.plot(timestamps, probs)
  plt.ylim(0, 1)
  path = os.path.join(outfolder, title + '.png')
  print('Timestamps being saved at ' + path)
  plt.savefig(path)


def show_frames(frames, probs, nskip=1, outfolder=None, title=None):
  print(f'Probability video is shouting: {round(np.mean([p for p in probs if p]), 2)}')
  probs = pd.Series(probs).ewm(span=10).mean().values
  numimgs = len(frames)
  ncols = 10
  nrows = np.ceil(numimgs/(ncols * nskip))
  plt.figure(figsize=(ncols, nrows))
  for i, (frame, prob) in enumerate(zip(frames[0:numimgs:nskip], probs[0:numimgs:nskip])):
    plt.subplot(nrows, ncols, i+1)
    plt.title(f'{round(prob, 2)}')
    plt.imshow(np.array(frame))
    plt.axis('off')
  plt.tight_layout()
  path = os.path.join(outfolder, title + '_frames.png')
  print('Frames being saved at ' + path)
  plt.savefig(path)


def get_frame_prob(faces):
  faceprobs = predict_face_probs([faces], resnet, pca, clf)
  return max(faceprobs) # return prob of at least one person shouting in video
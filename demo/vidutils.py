import cv2
from PIL import Image
import numpy as np
import os
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as preprocess_input19
from facenet_pytorch import MTCNN


# Load face detector
mtcnn = MTCNN(margin=30, keep_all=False, image_size=160).eval()
fmodel = VGG19(weights='imagenet', include_top=False)


class DetectionPipeline:
    """Pipeline class for detecting faces in the frames of a video file."""

    def __init__(self, detector, n_frames=None, batch_size=60, resize=0.25):
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
                    frame = frame.resize([int(d * self.resize)
                                         for d in frame.size])
                batch.append(frame)

                # When batch is full, detect faces and reset frame list
                if len(batch) % self.batch_size == 0 or j == sample[-1]:
                    faces.extend(self.detector(batch))
                    batch = []
                frames.append(frame)

        v_cap.release()

        return faces, frames, sample * (1 / fps)


# Define face detection pipeline
detection_pipeline = DetectionPipeline(detector=mtcnn, batch_size=60)


def scale01(x):
  return (x - x.min()) / (x.max() - x.min())


def isface(face):
    return face is not None


def get_features(imgs):
  imgs = preprocess_input19(imgs)
  return fmodel.predict(imgs).reshape(len(imgs), -1)


def singlepred(model, x):
    return model.predict(get_features(np.expand_dims(x, 0)))[0][0]


def get_face_probs(model, faces):
    return np.array([None if face is None else singlepred(model, face) for face in faces])


def load_video_frames(path, skip=1):
    faces, frames, timestamps = detection_pipeline(path)
    transform = lambda x: (
        scale01(x.permute(1, 2, 0).numpy()) * 255).astype(np.uint8)
    faces = [None if f is None else transform(f) for f in faces]
    return frames[::skip], faces[::skip], timestamps[::skip]

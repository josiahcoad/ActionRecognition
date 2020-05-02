from keras.applications.vgg19 import VGG19, preprocess_input
import librosa

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input,
              outputs=base_model.get_layer('flatten').output)


def get_features(imgarrays):
    imgs = []
    for imgarray in imgarrays:
        x = cv2.resize(imgarray, dsize=(224, 224),
                       interpolation=cv2.INTER_CUBIC)
        x = np.dstack((x, x, x))
        imgs.append(x)
    imgs = np.stack(imgs)
    imgs = preprocess_input(imgs)
    return model.predict(imgs)


def split_overlapping(a, size, step):
    return [a[i: i + size] for i in range(0, len(a), step)]


def get_audio_probs(clf, filename):
    y, sr = librosa.load(filename)
    test = []
    times = []
    step = sr  # full second step
    # split into 1 second (non-overlapping) chunks
    for i, chunk in enumerate(split_overlapping(y, sr, step)):
        times.append(i)
        test.append(librosa.feature.melspectrogram(y=chunk, sr=sr))

    return dict(zip(times, clf.predict_proba(get_features(test)).T[1]))

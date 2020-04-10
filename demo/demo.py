from utils import get_frame_probs, get_fps, mk_pred_plot, mk_pred_json
from tensorflow import keras
import numpy as np



def main(vidpath, resultspath):
    model = keras.models.load_model('models/keras_ffn_personal.h5')
    probs = get_frame_probs('openpose_json/personal/shout/1/*', model)
    fps = get_fps(vidpath)
    ts = fps * np.arange(len(probs))
    predictions = np.hstack(ts, probs)
    title = get_basename(vidpath)
    mk_pred_plot(predictions, resultspath, title)
    mk_pred_json(predictions, resultspath, title)



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
    main(vidpath, resultspath)

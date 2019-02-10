import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import librosa


def sample_wav_by_time(wav_path, frame_sec):
    """
    Samples audio by chunks of fixed time

    :param wav_path: path to wav file to sample
    :param frame_sec: int, length of each chunk in seconds
    :return: Pandas DataFrame, each chunk represented by row
    """
    rate, audio = wav.read(wav_path)
    frame_size = int(rate * frame_sec)
    frames = np.array([np.concatenate((np.mean(librosa.feature.melspectrogram(np.array(audio[i:i+frame_size], dtype=np.float32), rate), axis=0),
                                       np.mean(librosa.feature.mfcc(np.array(audio[i:i+frame_size], dtype=np.float32), rate), axis=0)))
                       for i in range(0, len(audio), frame_size)])
    return pd.DataFrame(frames)


def main():
    wav_path = "/home/kurbanov/Data/vocalizationcorpus/data/S0458.wav"
    df = sample_wav_by_time(wav_path, frame_sec=0.01)
    print(df.shape)
    print(df.head())


if __name__ == '__main__':
    main()

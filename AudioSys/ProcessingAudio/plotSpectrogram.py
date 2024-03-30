import wave
import librosa
from pydub import AudioSegment
from matplotlib import pyplot as plt
import numpy as np


def convert_audio_to_spectogram_log(filename):
    x, sr = librosa.load(filename, sr=32000)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    fig, ax = plt.subplots()
    #plt.figure(figsize=(14, 5))
    img = librosa.display.specshow(Xdb, ax = ax, x_axis = 'time', y_axis = 'log')
    font_dict = {'fontsize': 10, 'color': 'black', 'family': 'serif'}
    ax.set_title(f'Spectrogram with Logarithmic Frequency Y-Axis ({filename})', fontdict=font_dict)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    fig.savefig(f'{filename}.jpg')

def amplify(fileN, num):
    filenam = fileN + f'.wav'
    song = AudioSegment.from_mp3(filenam)
    louder_song = song + num  #num = 1
    names = fileN + f'_({num}).wav'
    louder_song.export(names, format='wav')
    convert_audio_to_spectogram_log(names)    

if __name__ == "__main__":
    #amplify('chunk_cat43', -60)
    #amplify('chunk_drone8', 0)

    convert_audio_to_spectogram_log('328_2244_4d0eefe9.wav')

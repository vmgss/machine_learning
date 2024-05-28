import librosa.display
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

if __name__ == '__main__':
    file="sample.wav"
    signal, sr = librosa.load(file)
    plt.figure(figsize = (14,5))
    librosa.display.waveshow(y = signal, sr= sr, color='blue')
    plt.show()
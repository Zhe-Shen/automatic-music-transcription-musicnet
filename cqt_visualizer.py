import h5py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys

def main(path):
    f = h5py.File(path, 'r')
    print(f['cqt'].shape)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(f['cqt']), x_axis='time', y_axis='cqt_note', \
                            sr=44100, ax=ax)
    plt.title('CQT spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
if __name__ == '__main__':
    # example: ./data/data_16K/test_data/1759.h5
    main(sys.argv[1])
import numpy as np
import librosa
from tqdm import tqdm

from pathlib import Path
from src.feature.spectrograms_inversion import *

class Mel_spectrogram_converter_inverter():
    def __init__(self):
        # ### Parameters ###
        self.signal_ratio=16000
        self.fft_size = 2048  # window size for the FFT
        self.step_size = self.fft_size // 16  # distance to slide along the window (in time)
        self.spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
        self.lowcut = 500  # Hz # Low cut for our butter bandpass filter
        self.highcut = 15000  # Hz # High cut for our butter bandpass filter
        # For mels
        self.n_mel_freq_components = 128  # number of mel frequency channels
        self.shorten_factor = 1  # how much should we compress the x-axis (time)
        self.start_freq = 300  # Hz # What frequency to start sampling our melS from
        self.end_freq = 8000  # Hz # What frequency to stop sampling our melS from

        self.mel_filter, self.mel_inversion_filter = create_mel_filter( fft_size=self.fft_size,
                                                                        n_freq_components=self.n_mel_freq_components,
                                                                        start_freq=self.start_freq,
                                                                        end_freq=self.end_freq,
                                                                        samplerate = self.signal_ratio)

    def invert_mel_spectrogram(self, mel_spec):
        mel_inverted_spectrogram = mel_to_spectrogram(
                                                        mel_spec,
                                                        self.mel_inversion_filter,
                                                        spec_thresh=self.spec_thresh,
                                                        shorten_factor=self.shorten_factor,
                                                    )
        inverted_mel_audio = invert_pretty_spectrogram(
                                                            np.transpose(mel_inverted_spectrogram),
                                                            fft_size=self.fft_size,
                                                            step_size=self.step_size,
                                                            log=True,
                                                            n_iter=10,
                                                        )
        return inverted_mel_audio



    def convert2mel_spectrogram(self, file_path):
        data, signal_ratio = librosa.load(file_path, sr=self.signal_ratio, mono=True)
        wav_spectrogram = pretty_spectrogram(
                                                data.astype("float64"),
                                                fft_size=self.fft_size,
                                                step_size=self.step_size,
                                                log=True,
                                                thresh=self.spec_thresh,
                                            )
        mel_spec = make_mel(wav_spectrogram, self.mel_filter, shorten_factor=self.shorten_factor)
        return mel_spec

    def convert_files(self, data_dir, save_dir):

        data_dir = Path(data_dir)
        files = list(data_dir.glob('**/*.mp3'))
        save_dir = Path(save_dir)

        for file in tqdm(files):
            try:
                mel_spec = self.convert2mel_spectrogram(file)
                filename = str(file.name).replace("wav", "npy")
                np.save(save_dir / filename, mel_spec)
            except:
                print(file)


if __name__ == '__main__':
    converter = Mel_spectrogram_converter_inverter()

    converter.convert_files("./data/piano", "./data/piano_npy")

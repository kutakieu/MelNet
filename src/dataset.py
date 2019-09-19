import numpy as np

from torch.utils.data import Dataset

from src.feature.mel_spec_conversion_class import Mel_spectrogram_converter_inverter

class AudioDataset(Dataset):
    def __init__(self, data_dir):

        self.files = list(data_dir.glob('**/*.npy'))
        self.mel_spec_converter = Mel_spectrogram_converter_inverter()


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx, process_raw_file=False):

        if process_raw_file:
            x = self.mel_spec_converter.convert2mel_spectrogram(self.files[idx])[:, :1000 + 1]
        else:
            x = np.load(self.files[idx])[:, :1000 + 1]
        # print(x.shape)
        # print(type(x))

        return x.astype(np.float32)

# if __name__ == '__main__':

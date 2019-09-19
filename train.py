import os
import sys
import argparse
from pathlib import Path

import librosa
import numpy as np

import torch
import torch.optim as optim

from src.dataset import AudioDataset
from src.MelNet import MelNet

def main(args):
    print(args)

    time_steps = 32
    num_mels = 128
    dims = 512
    n_layers = 5
    n_mixtures = 10

    model = MelNet(args.batch_size, time_steps, num_mels, dims, n_layers, n_mixtures)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)


    dataset_training = AudioDataset(Path(args.data_dir))
    print(len(dataset_training))
    dataloader_training = torch.utils.data.DataLoader(dataset_training, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        print("step1")
        break
        for i, batch in enumerate(dataloader_training):
            print("step2")
            print(batch.shape)
            for j in range(batch.shape[2] // time_steps):
                print("step3")
                sub_batch = batch[:, :, time_steps*j : time_steps*(j+1)].contiguous().view(args.batch_size, time_steps, num_mels)
                target_batch = batch[:, :, time_steps*j+1 : time_steps*(j+1)+1].contiguous().view(args.batch_size, time_steps, num_mels)

                print(sub_batch.shape)
                print("here")
                params = model(sub_batch)

                loss = model.loss(params, target_batch)[:, -1, :]
                loss = torch.mean(loss)
                print(loss.shape)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break

            break
        break

    model.eval()
    model.batch_size = 1
    model.time_step = 1
    output = np.zeros((num_mels, 200)).astype(np.float32)
    new_sample = torch.from_numpy(output[:, :1].reshape(1, 1, num_mels))
    print(new_sample.shape)
    with torch.no_grad():
        for i in range(200):
            params = model(new_sample.view(1, 1, num_mels))

            new_sample = model.sample(params)
            output[:, i] = new_sample.numpy()

            print(i)



def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="MelNet")

    # Data
    parser.add_argument('--batch_size', type=int, default=2, help='Number of instances per batch during training')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--debug', action="store_true", help='if it is in debug mode')
    parser.add_argument('--data_dir', type=str, default="./data/piano_npy", help='path to a directory stores audio dataset')

    return parser.parse_args(args)

if __name__ == '__main__':
    main(get_options())

##############################################################################
import os
import sys
# Import Project Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##############################################################################

import random
import itertools
import tensorflow as tf
import numpy as np
import collections
import pandas as pd
import enum
import functools
AUTOTUNE = tf.data.experimental.AUTOTUNE


"Class containing methods to manipulate audio data"
class AudioUtils:

    @staticmethod
    def read_audio(path, label):

        raw_audio = tf.io.read_file(path)
        waveform, _ = tf.audio.decode_wav(raw_audio)
        waveform = waveform[Ellipsis, 0]
        return waveform, label

    @staticmethod
    def read_tfrecord(serialized_example):

        feature_description = { "audio": tf.io.FixedLenFeature((), tf.string), "label": tf.io.FixedLenFeature((), tf.string)}
        example = tf.io.parse_single_example(serialized_example, feature_description)
        waveform = tf.io.parse_tensor(example["audio"], out_type = float)
        label = tf.io.parse_tensor(example["label"], out_type = float)
        return waveform, label

    @staticmethod
    def pad(waveform, time_window=2.0, sequence_length=16000):

        padding = tf.maximum(int(time_window*sequence_length) - tf.shape(waveform)[0], 0)
        left_pad = padding // 2
        right_pad = padding - left_pad
        return tf.pad(waveform, paddings=[[left_pad, right_pad]])

    @staticmethod
    def extract_window(waveform, time_window=2.0, seg_length=15690):

        waveform = AudioUtils.pad(waveform, time_window=time_window)
        return tf.image.random_crop(waveform, [int(time_window*seg_length)])

    @staticmethod
    def extract_windows(waveform, time_window=2.0, hop_time_window=1.0, seg_length=15690):

        waveform = AudioUtils.pad(waveform, time_window=time_window)
        num_samples_per_window=int(time_window*seg_length)
        num_hop_samples = int(hop_time_window*seg_length)
        return tf.signal.frame(waveform, frame_length=num_samples_per_window, frame_step=num_hop_samples, pad_end=False)

    @staticmethod
    def extract_spectrogram(waveform, sample_rate=16000, frame_length=400, frame_step=160,  fft_length=1024, n_mels=64, fmin=60.0, fmax=7800.0):

        # A x-point STFT with frames of x ms and x% overlap.
        stfts = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length) 
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        # log_mel_spectrograms = tf.math.log(mel_spectrograms + 0.01)
        mel_spectrograms = tf.clip_by_value(mel_spectrograms, clip_value_min=1e-5, clip_value_max=1e8)
        log_mel_spectrograms = tf.math.log(mel_spectrograms)

        return log_mel_spectrograms

    @staticmethod
    def extract_mfcc(log_mel_spectrogram, n_mfccs=13, deltas=[False,False], paddings=tf.constant([[1,0],[0,0]]), train=False):

        if train:
            # Extract MFCCs
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :n_mfccs]
            # Extract delta and delta_delta
            if deltas[0]:
                delta = mfccs - tf.pad(tensor=mfccs[:-1], paddings=paddings, mode="CONSTANT", constant_values=0)
                if deltas[1]:
                    d_delta = delta - tf.pad(tensor=delta[:-1], paddings=paddings, mode="CONSTANT", constant_values=0)
                    mfccs = tf.concat([mfccs, delta, d_delta], 1)
                else:
                    mfccs = tf.concat([mfccs, delta], 1)
        else:
            # Extract MFCCs
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :n_mfccs]
            # Extract delta and delta_delta
            if deltas[0]:
                pad = tf.pad(tensor=mfccs[:,:-1], paddings=paddings, mode="CONSTANT", constant_values=0)
                delta = mfccs - tf.pad(tensor=mfccs[:,:-1], paddings=paddings, mode="CONSTANT", constant_values=0)
                if deltas[1]:
                    d_delta = delta - tf.pad(tensor=delta[:,:-1], paddings=paddings, mode="CONSTANT", constant_values=0)
                    mfccs = tf.concat([mfccs, delta, d_delta], 2)
                else:
                    mfccs = tf.concat([mfccs, delta], 2)

        return mfccs

    @staticmethod
    def expand_axis(log_mel_spectrograms, label):
        return log_mel_spectrograms[Ellipsis, tf.newaxis], label

    @staticmethod
    def prepare_example_tw(waveform, label, time_window=2.0, n_mels=64, n_mfccs=None, deltas=[False,False]):

        # Pad waveform
        waveform = AudioUtils.pad(waveform)

        # L2 normalize the audio
        waveform = tf.math.l2_normalize(waveform, epsilon=1e-9)

        # Split into random time-window second segments
        waveform = AudioUtils.extract_windows(waveform, time_window=time_window, hop_time_window=1.0)

        # Extract spectrograms
        log_mel_spectrograms = tf.map_fn(fn=functools.partial(AudioUtils.extract_spectrogram, n_mels=n_mels, frame_length=400, fft_length=1024, frame_step=160), elems=waveform, fn_output_signature=(tf.float32))

        # Extract MFCCs
        if n_mfccs != None:
            mfcc = tf.map_fn(fn=functools.partial(AudioUtils.extract_mfcc, n_mfccs=n_mfccs, deltas=deltas, paddings=([[1,0],[0,0]]), train=True),elems=log_mel_spectrograms,fn_output_signature=(tf.float32))
            return mfcc, tf.repeat(input=label, repeats=tf.shape(mfcc)[0])

        return log_mel_spectrograms, tf.repeat(input=label, repeats=tf.shape(log_mel_spectrograms)[0])

    @staticmethod
    def prepare_example(waveform, label, time_window=2.0, n_mels=64, n_mfccs=None, deltas=[False,False], cast_to_float=False):

        # Normalizes audio: `int64` -> `float32`
        if cast_to_float:
            waveform = tf.cast(waveform, tf.float32) / float(tf.int16.max)

        # Pad waveform
        waveform = AudioUtils.pad(waveform)

        # L2 normalize the audio
        waveform = tf.math.l2_normalize(waveform, epsilon=1e-9)

        # Take random one second segment
        waveform = AudioUtils.extract_window(waveform, time_window=time_window)

        # Extract spectrograms
        log_mel_spectrograms = AudioUtils.extract_spectrogram(waveform, n_mels=n_mels)

        # Extract MFCCs
        if n_mfccs != None:
            mfcc = AudioUtils.extract_mfcc(log_mel_spectrogram=log_mel_spectrograms, n_mfccs=n_mfccs, deltas=deltas, paddings=([[1,0],[0,0]]), train=True)
            return mfcc, label

        return log_mel_spectrograms, label
    
    @staticmethod
    def prepare_test_example(waveform, label, n_mels=64, n_mfccs=None, deltas=[False,False], cast_to_float=False):

        # Normalizes audio: `int64` -> `float32`
        if cast_to_float:
            waveform = tf.cast(waveform, tf.float32) / float(tf.int16.max)
        # Pad waveform
        waveform = AudioUtils.pad(waveform)
        # Chunk waveform
        waveform = tf.signal.frame(waveform, frame_length=20 * 1569, frame_step=4 * 1569, pad_end=False)
        # L2 normalize the audio
        waveform = tf.math.l2_normalize(waveform, axis=-1, epsilon=1e-9)
        # Extract spectrograms
        log_mel_spectrogram = AudioUtils.extract_spectrogram(waveform, n_mels=n_mels, frame_length=400, fft_length=1024, frame_step=160)
        # Extract MFCCs
        if n_mfccs != None:
            mfcc = AudioUtils.extract_mfcc(log_mel_spectrogram=log_mel_spectrogram, n_mfccs=n_mfccs, deltas=deltas, paddings=([[0,0],[1,0],[0,0]]))
            return mfcc, label

        return log_mel_spectrogram, label

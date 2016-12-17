import fnmatch
import os
import re
import threading

import librosa
import numpy as np
import tensorflow as tf


def find_files(file_list, audio_directory, label_directory, audio_ext='*.wav', label_ext='.lab'):
    
    try: 
        fid = open(file_list, 'r')
        filenames = fid.readlines()
        fid.close()
    except IOError:
        return None

    audio_fullpathnames = []
    label_fullpathnames = []
    for filename in filenames:
        audio_fullpathnames.append(audio_directory + filename.rstrip() + audio_ext)
        label_fullpathnames.append(label_directory + filename.rstrip() + label_ext)   

    return audio_fullpathnames, label_fullpathnames


def load_generic_audio_label_old(file_list, audio_directory, label_directory, labels_dim, audio_ext='.wav', label_ext='.lab', sample_rate=16000, frame_shift=0.005):
    '''Generator that yields audio waveforms from the directory.'''
    audio_files, label_files = find_files(file_list, audio_directory, label_directory, audio_ext, label_ext)
    for audio_filename, label_filename in zip(audio_files, label_files):
        audio, _ = librosa.load(audio_filename, sr=sample_rate, mono=True)
        
        with open(label_filename, 'rb') as fid:
            labels = np.fromfile(fid, dtype=np.float32, count=-1)
        fid.close()
             
        n_frames = len(labels)/labels_dim 
        labels = labels.reshape((n_frames, labels_dim))
       
        samples_per_frame = sample_rate*frame_shift
        labels = np.repeat(labels, samples_per_frame, 0) # upsample

        n_audio_samples = len(audio)
        n_label_samples = labels.shape[0]

        if (n_audio_samples > n_label_samples):
            audio = audio[0:n_label_samples]
        elif (n_audio_samples < n_label_samples):
            labels = labels[0:n_audio_samples, :]   

        audio = audio.reshape(-1, 1) 
 
        yield audio, labels, audio_filename


def load_generic_audio_label(file_list, audio_directory, label_directory, labels_dim, audio_ext='.wav', label_ext='.lab', sample_rate=16000, frame_shift=0.005):
    '''Generator that yields audio waveforms from the directory.'''
    audio_files, label_files = find_files(file_list, audio_directory, label_directory, audio_ext, label_ext)
    for audio_filename, label_filename in zip(audio_files, label_files):
        audio, _ = librosa.load(audio_filename, sr=sample_rate, mono=True)
        
        n_audio_samples = len(audio) 

        with open(label_filename, 'rb') as fid:
            labels = np.fromfile(fid, dtype=np.float32, count=-1)
        fid.close()
             
        n_frames = len(labels)/labels_dim 
        labels = labels.reshape((n_frames, labels_dim))
       
        frame_length = 0.025 
        samples_per_frame = int(sample_rate*frame_length) 
        samples_per_shift = int(sample_rate*frame_shift)

        # Upsample labels. There is 80% overlap between frames. Each frame is 0.025 sec. The frame shift is 0.005 sec
        # ---------- frame_0 
        #   ---------- frame_1
        #     ---------- frame_2
        #       ---------- frame_3
        #         ---------- frame_4
        # The following overlap-and-add procedure may not be appropriate for features 417:425 
        n_label_samples = int((n_frames + frame_length/frame_shift - 1)*samples_per_shift)
        upsampled_labels = np.zeros((n_label_samples, labels_dim))
        count = np.zeros((n_label_samples, ), dtype=np.float32)

        for j in range(n_frames):
            i = j*samples_per_shift
            count[i:i+samples_per_frame] += 1
            for k in range(samples_per_frame):
                upsampled_labels[i+k, :] += labels[j, :]

        for j in range(n_label_samples):
            upsampled_labels[j, :] /= count[j]    

        if (n_audio_samples > n_label_samples):
            audio = audio[0:n_label_samples]
        elif (n_audio_samples < n_label_samples):
            labels = labels[0:n_audio_samples, :]   

        audio = audio.reshape(-1, 1) 
 
        yield audio, upsampled_labels, audio_filename



def trim_silence(audio, label, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    if indices.size:
        return audio[indices[0]:indices[-1]], label[indices[0]:indices[-1], :]
    else: 
        audio[0:0], label[0:0, :]



class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord, 
                 file_list,
                 audio_dir,
                 label_dir,
                 label_dim,
                 audio_ext,
                 label_ext,
                 sample_rate,
                 frame_shift,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=256):
        self.coord = coord
        self.file_list = file_list
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.label_dim = label_dim
        self.audio_ext = audio_ext
        self.label_ext = label_ext  
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift   
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.threads = []
        self.audio_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.FIFOQueue(queue_size, ['float32', 'float32'])
        self.enqueue = self.queue.enqueue([self.audio_placeholder, self.label_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(file_list, audio_dir, label_dir, audio_ext, label_ext):
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

        

    def dequeue(self):
        output = self.queue.dequeue()
        return output

    def thread_main(self, sess):
        audio_buffer_ = np.array([], dtype=np.float32)
        label_buffer_ = np.zeros((0, self.label_dim), dtype=np.float32)
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio_label(self.file_list, self.audio_dir, self.label_dir, self.label_dim, self.audio_ext, self.label_ext, self.sample_rate, self.frame_shift)
            for audio, labels, filename in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio, labels = trim_silence(audio[:, 0], labels, self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    audio_buffer_ = np.append(audio_buffer_, audio, axis=0)
                    label_buffer_ = np.append(label_buffer_, labels, axis=0)
                    while len(audio_buffer_) > self.sample_size:
                        audio_piece = audio_buffer_[:self.sample_size, :]
                        label_piece = label_buffer_[:self.sample_size, :]
                        sess.run(self.enqueue,
                                 feed_dict={self.audio_placeholder: audio_piece, self.label_placeholdes: label_piece})
                        audio_buffer_ = audio_buffer_[self.sample_size:]
                        label_buffer_ = label_buffer_[self.sample_size:] 
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.audio_placeholder: audio, self.label_placeholder: labels})
                    
                    
                    

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

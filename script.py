import matlab.engine
import os
import ntpath
from shutil import move
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from scipy import signal
from scipy import stats
from matplotlib import rcParams
import array
import math
import numpy as np
import random
import wave

rcParams['image.cmap']= 'Greys'
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def getResults():

    wavlocation = os.path.expanduser("~/Desktop/idbm")
    cleansound_fileswav = [f for f in os.listdir(wavlocation) if f.endswith('_60dB.wav')]
    noisesound_fileswav = [f for f in os.listdir(wavlocation) if f.endswith('_noise.wav')]
    x = 0
    while x < len(cleansound_fileswav):
        y = 0
        clean = ntpath.basename(cleansound_fileswav[x])
        while y <len (noisesound_fileswav):
            #opens all files
            fs4f1,f1 = scipy.io.wavfile.read(cleansound_fileswav[x])
            fs4f2,f2 = scipy.io.wavfile.read(noisesound_fileswav[y])
            #strips the name for saving purposes
            cleanStripped = cleansound_fileswav[x].strip(".wav")
            noiseStripped=noisesound_fileswav[y].strip(".wav") 
            # SNR = 20 ... failed attempt from code underneath
            # changedSNR = cleanStripped +"+" + noiseStripped + str(SNR) ... not currently ussed
            l = min(len(f1),len(f2))
            #f3 = signaltoSNR(f1[:l],f2[:l],changedSNR, SNR) ... same with this line
            f3 = f1[:l] + f2[:l]
            ff1,tt1,f1_stft = scipy.signal.stft(f1[:l].astype(float),fs=fs4f1, nperseg =1024, noverlap=256)
            ff2,tt2,f2_stft = scipy.signal.stft(f2[:l].astype(float),fs= fs4f2, nperseg =1024, noverlap=256)
            ff3,tt3,f3_stft = scipy.signal.stft(f3.astype(float),fs=fs4f1, nperseg =1024, noverlap=256)
            mask = abs(f1_stft)>abs(f2_stft)

            f4 = mask * f3_stft
            _,f4 = scipy.signal.istft(f4.astype(float),nperseg =1024, noverlap=256)
            f4 = f4.real

            scipy.io.wavfile.write(cleanStripped+"+"+noiseStripped+".wav",44100,f4/max(abs(f4)))
            fig = plt.figure()
            q1 = fig.add_subplot(2,2,1)
            plt.pcolormesh(tt1,ff1,abs(f1_stft)**0.5)
            colorbar()
            tight_layout()
            plt.title ("Mask")
            plt.xlabel("Time (secs)")
            plt.ylabel("Frequency (Hz)")
            q2 = fig.add_subplot(2,2,2)
            plt.pcolormesh(tt2,ff2,abs(f2_stft)**0.5)
            colorbar()
            tight_layout()
            plt.title ("Mask")
            plt.xlabel("Time (secs)")
            plt.ylabel("Frequency (Hz)")
            q3 = fig.add_subplot(2,2,3)
            plt.pcolormesh(tt3,ff3,abs(f3_stft)**0.5)
            colorbar()
            tight_layout()
            q1 = fig.add_subplot(2,2,4)
            plt.pcolormesh(tt3,ff3,mask.astype(float64))
            colorbar()
            tight_layout()
            plt.title ("Mask")
            plt.xlabel("Time (secs)")
            plt.ylabel("Frequency (Hz)")
            plt.show()
            fig.savefig("full_figure.png")
            y += 1

        x += 1


#attempt at getting an snr
""" def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def cal_amp(cleanfile):
    n_samples = wave.readframes (cleanfile)
    np_fft = np.fft.fft(cleanfile)
    amplitude = 2/n_samples * np.abs(np_fft)
    return amplitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))
def signaltoSNR(cleanfile, noisefile, outnoisy, snr):
    
    clean_amp = cal_amp(cleanfile)
    noise_amp = cal_amp(noisefile)

    start = random.randint(0, len(noise_amp)-len(clean_amp))
    clean_rms = cal_rms(clean_amp)
    split_noise_amp = noise_amp[start: start + len(clean_amp)]
    noise_rms = cal_rms(split_noise_amp)

    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    
    adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms) 
    mixed_amp = (clean_amp + adjusted_noise_amp)

    if (mixed_amp.max(axis=0) > 44100): 
        mixed_amp = mixed_amp * (44100/mixed_amp.max(axis=0))
        clean_amp = clean_amp * (44100/mixed_amp.max(axis=0))
        adjusted_noise_amp = adjusted_noise_amp * (44100/mixed_amp.max(axis=0))

    noisy_wave = wave.Wave_write(outnoisy)
    noisy_wave.setparams(clean_wav.getparams())
    noisy_wave.writeframes(array.array('h', mixed_amp.astype(np.int16)).tostring() )
    noisy_wave.close() """

getResults()
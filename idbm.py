import matplotlib.pyplot as plt
import librosa 
import librosa.display
import ntpath
import os
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import tkinter as tk
import tkinter.filedialog as tkfile
import ntpath
import glob, os
from scipy.io.wavfile import read as read_wav
import librosa
import soundfile
from scipy.io.wavfile import read as read_wav
import numpy as np
from scipy.io import wavfile
from scipy import interpolate
import gc
import shutil
import wave
from shutil import move
import matlab.engine




def main():
    mp3conversion()
    makemono()
    getCorrectSR()

def getCorrectSR():
    newfolder = os.path.expanduser("~/Desktop/parismod/")
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    os.chdir(newfolder)
    folderparismod = os.path.expanduser("~/Desktop/parismod/")
    sound_fileswav = [f for f in os.listdir(folderparismod) if f.endswith('.wav')]
    print (sound_fileswav)
    x = 0
    while x < len (sound_fileswav):
        filename = ntpath.basename(sound_fileswav[x])
        NEW_SAMPLERATE = 44100

        old_samplerate, old_audio = wavfile.read(filename)

        if old_samplerate != NEW_SAMPLERATE:
            duration = old_audio.shape[0] / old_samplerate

            time_old  = np.linspace(0, duration, old_audio.shape[0])
            time_new  = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))

            interpolator = interpolate.interp1d(time_old, old_audio.T)
            new_audio = interpolator(time_new).T

            wavfile.write(filename, NEW_SAMPLERATE, np.round(new_audio).astype(old_audio.dtype))
        original = os.path.expanduser("~/Desktop/parismod/"+filename)
        new = os.path.expanduser("~/Desktop/idbm/"+filename)
        move(original, new)
        x += 1
        
        print ("Completed:" + filename)
    

    
def makemono():
    newfolder = os.path.expanduser("~/Desktop/parismod/")
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    folderparismod = os.path.expanduser("~/Desktop/parismod/")
    sound_fileswav = [f for f in os.listdir(folderparismod) if f.endswith('.wav')]
    x = 0
    while x < len (sound_fileswav):
        filename = ntpath.basename(sound_fileswav[x])
        mysound = AudioSegment.from_wav(folderparismod +"/" +filename)
        mysound = mysound.set_channels(1)
        mysound.export(folderparismod +"/" +filename, format="wav")
        x +=1

# def stftwavish ():

#     folderparismod = os.path.expanduser("~/Desktop/parismod/")
#     sound_fileswav = [f for f in os.listdir(folderparismod) if f.endswith('.wav')]
#     x = 0
#     while x < len(sound_fileswav) :
#         filename = ntpath.basename(sound_fileswav[x])
#         filenamestripped = filename.strip("_new.wav") 
#         print (filename)
#         locationoffile = folderparismod + "/" + filename
#         print (locationoffile)
#         y, sr = librosa.load(locationoffile)
#         D = np.abs(librosa.stft(y))
#         librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')
#         plt.title(filenamestripped)
#         plt.colorbar(format='%+2.0f dB')
#         plt.tight_layout()
#         print ("next graph queen")
#         x +=1
#         if x == len(sound_fileswav): 
#             break
#         plt.figure()
#     plt.show()


def mp3conversion():
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop/originalparis') 
    sound_filesmp3 = [f for f in os.listdir(desktop) if f.endswith('.mp3')]
    os.chdir(desktop)
    i = 0
    while i < len(sound_filesmp3) : 
        filename = ntpath.basename(sound_filesmp3[i])
        print (filename)
        filenamestripped = filename.strip(".mp3")
        sound = AudioSegment.from_mp3(filename)
        sound.export(filenamestripped + '_new' + ".wav", format="wav")
        i +=1

    x = 0
    sound_fileswav =[f for f in os.listdir(desktop) if f.endswith('.wav')]

    while x < len(sound_fileswav) : 
        filename = ntpath.basename(sound_fileswav[x])
        print (filename)
        filenamestripped = filename.strip(".wav")
        if "_new" in filename:
            original = os.path.expanduser("~/Desktop/originalparis/"+filename)
            new = os.path.expanduser("~/Desktop/parismod/"+filename)
            newfolder = os.path.expanduser("~/Desktop/parismod/")
            move(original, new)
        x +=1

    print ("Process has been completed")
main()
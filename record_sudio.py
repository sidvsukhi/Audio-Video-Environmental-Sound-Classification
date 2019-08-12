# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:06:14 2019

@author: siddh
"""

import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

from keras.models import model_from_json
import librosa
import numpy as np
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#loaded_model.summary()

testfile="S:/ty sem2/13-62-EN50/ESC-50-master/output.wav"
twave, tsr= librosa.load(testfile)
tmfccs = librosa.feature.mfcc(y=twave, sr=tsr, n_mfcc=40)
X = np.array(tmfccs.tolist())
num_rows = 72
num_columns = 120
num_channels = 1
X= X.reshape(1, num_rows, num_columns, num_channels)
output = loaded_model.predict(X)

sound_event=["airplane","breathing","brushing_teeth","can_opening","car_horn","cat","chainsaw",
             "chirping_bird","church_bells","clapping","clock_alarm","clock_tick","coughing",
             "cow","crackling_fire","crickets","crow","crying_baby","dog","door_wood_creaks",
             "door_wood_knocks","drinking_sipping","engine","fireworks","footsteps","frog",
             "glass_breaking","hand_saw","helicopter","hen","insects","keyboard_typing","laughing",
             "mouse_click","pig","pouring_water","rain","rooster","sea_waves","sheep","siren","sneezing",
             "snoring","thunderstorm","toiletflush","train","vaccum_cleaner","washing_machine",
             "water_drops","wind"]

rdict=dict(zip(sound_event,output[0]))
sorted(rdict.items(), key =lambda kv:(kv[1], kv[0]))

for k, v in rdict.items():
    if(v>0.00001):
        print(k, v*100)
#result = loaded_model.predict_classes(X)
#print(sound_event[result[0]])
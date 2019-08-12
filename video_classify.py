# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:28:18 2019

@author: siddh
"""

import urllib.request
import urllib.error
import re
import sys
import time
import os
import pipes

def video_to_audio(fileName):
	try:
		file, file_extension = os.path.splitext(fileName)
		file = pipes.quote(file)
		video_to_wav = 'ffmpeg -i ' + file + file_extension + ' ' + 'Sample' + '.wav'
      #video_to_wav = 'ffmpeg -i ' + file + file_extension + ' ' + file + '.wav'
		#final_audio = 'lame '+ file + '.wav' + ' ' + file + '.mp3'
		os.system(video_to_wav)
		#os.system(final_audio)
		#file=pipes.quote(file)
		#os.remove(file + '.wav')
		print("sucessfully converted ", fileName, " into audio!")
	except OSError as err:
		print(err.reason)
		exit(1)

def main():
	if len(sys.argv) <1 or len(sys.argv) > 2:
		print('command usage: python3 video_to_audio.py FileName')
		exit(1)
	else:
		filePath = sys.argv[1]
		# check if the specified file exists or not
		try:
			if os.path.exists(filePath):
				print("file found!")
		except OSError as err:
			print(err.reason)
			exit(1)
		# convert video to audio
		video_to_audio(filePath)
		time.sleep(1)
		
# install ffmpeg and/or lame if you get an error saying that the program is currently not installed 
if __name__ == '__main__':
	main()

from pydub import AudioSegment
t1 = 5 * 1000 #Works in milliseconds
newAudio = AudioSegment.from_wav("Sample.wav")
newAudio = newAudio[:t1]
newAudio.export('Sample.wav', format="wav") #Exports to a wav file in the current path.

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

testfile="S:/ty sem2/13-62-EN50/ESC-50-master/Sample.wav"
twave, tsr= librosa.load(testfile)
tmfccs = librosa.feature.mfcc(y=twave, sr=tsr, n_mfcc=40)
X = np.array(tmfccs.tolist())
num_rows = 72
num_columns = 120
num_channels = 1
X= X.reshape(1, num_rows, num_columns, num_channels)
output = loaded_model.predict(X)

sound_event=["airplane","breathing","brushing_teeth","can_opening","car_horn","cat","chainsaw",
"chirping_birds","church_bells","clapping","clock_alarm","clock_tick","coughing","cow","crackling_fire",
"crickets","crow","crying_baby","dog","door_wood_creaks","door_wood_knock","drinking_sipping",
"engine","fireworks","footsteps","frog","glass_breaking","hand_saw","helicopter","hen","insects",
"keyboard_typing","laughing","mouse_click","pig","pouring_water","rain","rooster","sea_waves",
"sheep","siren","sneezing","snoring","thunderstorm","toilet_flush","train","vacuum_cleaner",
"washing_machine","water_drops","wind"]


rdict=dict(zip(sound_event,output[0]))
sorted(rdict.items(), key =lambda kv:(kv[1], kv[0]))

#for k, v in rdict.items():
#    if(v>0.00001):
#        print(k, v*100)
#result = loaded_model.predict_classes(X)
#print(sound_event[result[0]])
from tkinter import Tk, Label, Button

def printSomething():
    for k, v in rdict.items():
        if(v>0.001):
            #print(k, v*100)
            p=str(k)+'      '+str(v*100)
    # if you want the button to disappear:
    # button.destroy() or button.pack_forget()
            label = Label(root, text=p ,padx=5,pady=5,font=('Montserrat',20))
            #label.config(width=200)
    #this creates a new label to the GUI
            label.pack() 

root = Tk()
button = Button(root, text="Predict", command=printSomething) 
button.pack()

root.mainloop()

os.remove('S:/ty sem2/13-62-EN50/ESC-50-master/Sample.wav')
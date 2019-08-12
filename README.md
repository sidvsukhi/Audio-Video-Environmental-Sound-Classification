# Audio-Video-Environmental-Sound-Classification
A sound classification technique using 5 layer neural network that is 4 CNN and 1 Dense layer.

How to use this repository?
1. Dataset- ESC-50 Environmental Sound
  The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
  link-https://github.com/karoldvl/ESC-50/archive/master.zip <br/>
  Dataset distribution <br/> 
  ![dataset distribution](https://github.com/sidvsukhi/Audio-Video-Environmental-Sound-Classification/blob/master/after%20noise%20reduction%20pie%20chart.png) <br/>
2. mytry.py- <br/>
The cardinal file which does all the preprocessing on the dataset including noise reduction, channel proessing as well as mfcc (mel frequency ceptral coefficient) generation. This file also builds the 5 layer neural network and trains it. It is trained using the mfcc of the sounds. Finally, it makes a .json file extension trained model in the folder.
The MFCC of dataset <br/>
  ![dataset mfcc](https://github.com/sidvsukhi/Audio-Video-Environmental-Sound-Classification/blob/master/mfcc.png) </br>
3. record_sudio.py- <br/>
This is an application file that records 5 sec audio from your device and using our trained model predicts the sound and gives you the prediction. <br/>
4. video_classify.py- <br/>
This is also an application file that initially converts the video to audio, and then predicts the sound based on first 5 sec of the sound of audio file using out trained classifier. <br/>
![video_to_audio](https://github.com/sidvsukhi/Audio-Video-Environmental-Sound-Classification/blob/master/video%20to%20audio.jpg) <br/>

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
  ![dataset mfcc](https://github.com/karoldvl/ESC-50/blob/master/esc50.gif) </br>
3. record_sudio.py- <br/>
This is an application file that records 5 sec audio from your device and using our trained model predicts the sound and gives you the prediction. <br/>
4. video_classify.py- <br/>
This is also an application file that initially converts the video to audio, and then predicts the sound based on first 5 sec of the sound of audio file using out trained classifier. <br/>
![video_predict](https://github.com/sidvsukhi/Audio-Video-Environmental-Sound-Classification/blob/master/model%20layers.JPG) <br/><br/>

*For executing file <br/>
1.In mytry.py <br/>
	Set csv file path to esc50.csv file <br/>
	Set subpath to inside audio (eg- E:/projects/Master/audio/) <br/>
	Run the mytry.py file now as python mytry.py <br/>
	Model .json file will be stored in Master as "model1.json" and weights as "model1.h5" <br/> <br/>
2.In video_classify.py <br/>
	Set testfile path to Sample.wav inside Master <br/>
	(Sample.wav will be the .wav file converted from video to audio eg- E:/projects/Master/Sample.wav) <br/>
	Set os.remove to Sample.wav path (same as above) <br/>
	Sun the video classification file as <br/>
	python video_classify.py Farm_Animal.mp4 <br/>

*Note- classification applies to only first 5 seconds of the video/audio <br/>
![video_to_audio](https://github.com/sidvsukhi/Audio-Video-Environmental-Sound-Classification/blob/master/video%20to%20audio.JPG) <br/>
![video_predict](https://github.com/sidvsukhi/Audio-Video-Environmental-Sound-Classification/blob/master/predict.JPG) <br/><br/>

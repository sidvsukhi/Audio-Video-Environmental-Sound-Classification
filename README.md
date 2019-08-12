# Audio-Video-Environmental-Sound-Classification
A sound classification technique using 5 layer neural network that is 4 CNN and 1 Dense layer.

How to use this repository?
1. Dataset- ESC-50 Environmental Sound
  The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
  link-https://github.com/karoldvl/ESC-50/archive/master.zip <br/>
  Dataset distribution <br/> 
  ![dataset distribution]() <br/>
2. mytry.py- <br/>
The cardinal file which does all the preprocessing on the dataset including noise reduction, channel proessing as well as mfcc (mel frequency ceptral coefficient) generation. This file also builds the 5 layer neural network and trains it. It is trained using the mfcc of the sounds. Finally, it makes a .json file extension trained model in the folder.
The MFCC of dataset <br/>
  ![dataset mfcc](https://github.com/karoldvl/ESC-50/blob/master/esc50.gif) </br>

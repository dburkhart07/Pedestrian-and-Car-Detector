# Pedestrian-and-Car-Detector
This app is able to take any video or image containing pedestrians and cars, and highlights them in a rectangle. For this app, I used the opencv Python library, as well as a pre-trained car detector model and haar cascade model to detect pedestrians, both of which attached in the files list. The program assigns any image or video that is saved in the environment to static variables, and applies the car and pedestrian detectors on them, producing a list of the coordinates of each pedestrian and car on the screen. We then use those coordinates to create a rectangle around each of those cars and pedestrians. For videos, we continue to loop through this entire process while the video successfully reads frames. This means that for the conversion of image to video, we just treat the video as a bunch of images that we are detecting pedestrians and cars, and continue to loop through that until the video no longer reads any frames (the video ends). The detector can run until the video ends, or until the space bar is pressed.
Credit to @Clever Programmer on Youtube 



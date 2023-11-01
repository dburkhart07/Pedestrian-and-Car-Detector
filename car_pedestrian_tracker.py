import cv2

#Static variables here:

#Car image/Video
image_file = 'car.jpg'
video = cv2.VideoCapture('car_video1.mp4')

#Pre-trained classifiers
classifier_file = 'car_detector.xml'
pedestrian_classifier = 'haarcascade_fullbody.xml'

#Object detection here:

#create the classifiers
car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)


#Run forever until some input or video stops
while True:

    #Read current frame
    (read_frame, frame) = video.read()

    #Make sure read_frame was successful
    if read_frame:

        #Convert to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        break
    
    #Detect where the cars are in the image
    detected_cars = car_tracker.detectMultiScale(grayscale_frame)
    detected_pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

    #Draw rectangles around the cars
    for(x, y, w, h) in detected_cars:
        #                                        B  G  R
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

    #Draw rectangles around the pedestrians
    for(x, y, w, h) in detected_pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 255), 2)

    #Dislpay image with the ojbects spotted
    cv2.imshow('Cool Car Pedestrian Detector', frame)

    # Continue in the video until a key is pressed
    key = cv2.waitKey(1)

    #Stop is space bar is pressed
    if key == 32:
        break

#Release video capture object
video.release()



'''
Use for detecting an cars in an image

#Opencv image
image = cv2.imread(image_file)

#grayscale it (for haar cascade)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#create the car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#Detect where the cars are in the image
detected_cars = car_tracker.detectMultiScale(grayscale_image)

#Draw rectangles around the cars
for(x, y, w, h) in detected_cars:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)

#Dislpay image with cars spotted
cv2.imshow('Super cool image', image)

#Keep image up until key is pressed
cv2.waitKey()
'''


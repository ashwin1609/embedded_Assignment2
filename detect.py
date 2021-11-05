
import time
import numpy as np
import cv2   

# Loading the classifier needed for the detection
car_classifier = cv2.CascadeClassifier('cars.xml')
bike_classifier = cv2.CascadeClassifier('bikes.xml')

# setting the time method in python
start = time.time()
# Loads the input video
input_video = cv2.VideoCapture('cars.mp4')
end = time.time()

# while loop to read the video frame by frame
while input_video.isOpened():
        

    ret, frame = input_video.read()
    # converting the frames to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # the parameters of detectMultiScale are scale factor and minimum neighbors
    cars = car_classifier.detectMultiScale(gray, 1.1, 2)
    bikes = bike_classifier.detectMultiScale(gray,1.1,1)
    
# for loops to label and draw boxes around the detected object
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        img_gray = gray[y:y+h, x:x+w]
        cars = car_classifier.detectMultiScale(img_gray)
        cv2.putText(frame, 'car', (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    for (x,y,w,h) in bikes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        image_gray = gray[y:y+h, x:x+w]
        bikes = bike_classifier.detectMultiScale(image_gray)
        cv2.putText(frame, 'bike', (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# displays the video frame
    cv2.imshow("Vehicle detected on the road", frame)

# Key to stop the code and exit the frame
# It waits for every millisecond until the 'x' key is pressed to exit the code    
    if cv2.waitKey(1) == ord('x') : 
        break

# Once the code stops running it destroys the frame window
input_video.release()
cv2.destroyAllWindows()

# Calculates the time taken to read the file
diff = end - start
print(diff)
import sys
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

if len(sys.argv)==1:
    args={
    "prototxt":"MobileNetSSD_deploy.prototxt.txt",
    "model":"MobileNetSSD_deploy.caffemodel",
    "confidence":0.2,
    }
else:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())



# ModelNet SSD Object list init
CLASSES = ["arriere-plan", "avion", "velo", "oiseau", "bateau",
    "bouteille", "autobus", "voiture", "chat", "chaise", "vache", "table",
    "chien", "cheval", "moto", "personne", "plante en pot", "mouton",
    "sofa", "train", "moniteur"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load model file
print("Load Neural Network...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

if __name__ == '__main__':

    print("Start ...")
    # vc = cv2.VideoCapture('./img/video.mp4') # from video

    time.sleep(2.0)
    fps = FPS().start()
    
    while True:
        frame= cv2.imread('./img/pdog.jpg') # from image file
        #ret, frame=vc.read() #from video
        frame = imutils.resize(frame, width=800)

        # Create blob from image
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Feed input to neural network 
        net.setInput(blob)
        detections = net.forward()

        # Detection loop
        for i in np.arange(0, detections.shape[2]):
            # Compute Object detection probability
            confidence = detections[0, 0, i, 2]
            
            # Suppress low probability
            if confidence > args["confidence"]:
                # Get index and position of detected object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Create box and label
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    
                # enregistrement de l'image détectée 
                cv2.imwrite("detection.png", frame)
                
                
        # Show video frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit script with letter q
        if key == ord("q"):
            break

        # FPS update 
        fps.update()

    # Stops fps and display info
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
    vc.release()
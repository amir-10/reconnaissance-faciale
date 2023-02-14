import cv2
import sys
import os

def extractFace(path,name):
    # os.system('mv batouche.jpg db/')

    # imagePath = sys.argv[1]
    imagePath = path
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2 — Writing and Running the Face Detector Script
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
    ) 

    print("Found {0} Faces!".format(len(faces)))
    if len(faces)>0:
        max=0
        for (x, y, w, h) in faces:
            #max surface
            if (x+w)*(y+h) > max:
                max = (x+w)*(y+h)
                roi_color = image[y:y + h, x:x + w] 
                print("[INFO] Object found. Saving locally.") 
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            

        #save the nearest face
        status = cv2.imwrite(str(name)+'.jpg', roi_color)
        os.system('mv '+str(name)+'.jpg faces/')

        # status = cv2.imwrite('faces_detected.jpg', image)
        print ("Image "+str(name)+".jpg written to filesystem: ",status)


def AreFaces(img):

    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
    ) 

    print("[AreFaces] Found {0} Faces!".format(len(faces)))
    if len(faces)>0:
        return True
    return False
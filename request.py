import requests
from WebCam_Capture import takePic
from face_extraction import extractFace, AreFaces
import cv2
import sys
name = sys.argv[1]

# url = 'http://localhost:3000/predict'

# r = requests.post(url,json={'demo-age' : 38,
#                             'demo-year': 1999,
#                             'demo-axi' : 3
#                             }
#                  )


def newUser(name):
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(2)
    j = 70
    while True and j != 0:
        j = j-1

        check, frame = webcam.read()
        # print(check) #prints true as long as the webcam is running
        # print(frame) #prints matrix values of each framecd
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if j == 0:
            img_path = 'primaryDB/'+str(name)+'.jpg'
            cv2.imwrite(filename=img_path, img=frame)
            if AreFaces(img_path) == True:
                webcam.release()
                # img_new = cv2.imread('primaryDB/oussama.jpg', cv2.IMREAD_GRAYSCALE)
                # img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                # cv2.destroyAllWindows()
                # Frame.show()
                print("Image saved!")

                extractFace(img_path, name)
                cv2.destroyWindow("Capturing")

                break
            else:
                os.system('rm '+img_path)
                print('[newUser] Image deleted!')
                webcam.release()
                cv2.destroyWindow("Capturing")


newUser(name)
# print(r.json())
# print(r.json()[0]["name"])

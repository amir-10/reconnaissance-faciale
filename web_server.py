from compare import final_compare_blink
import dlib
from detect_blinks import check_extract_faces, eye_aspect_ratio
import time
from flask import Flask, request, render_template, Response  # loading in Flask
import json
import os
import subprocess
from subprocess import PIPE, Popen
import cv2
from imutils import face_utils
import imutils
from imutils.video.videostream import VideoStream
import numpy as np


def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]


# creating a Flask application
app = Flask(__name__, template_folder="template")


# creating user url and only allowing post requests.
@app.route('/newUser', methods=['GET', 'POST'])
def new():
    if request.method == 'POST':
        os.system('python3 request.py '+str(request.form.get('name')))

        return render_template('index.html', status="User Created")

    return render_template('index.html', status='')


@app.route('/', methods=['GET'])
def inex():
    return render_template('index.html', status='')


@app.route('/auth', methods=['POST'])
def auth():
    print(" ---------- [AUTH-START] ---------- ")

    user_name = request.form.get('name')
    out = subprocess.run(['python3', 'compare.py', '-u',
                         user_name], stdout=subprocess.PIPE)
    # print("OUTPUT ------ ")
    # print(out)
    result = True  # str(out).split('###')[7]  # [1] is MSE , [3],TRUE OR FALSE
    print(" RESULT ----- ")
    print("-", result, "-")

    print(" ---------- [AUTH-END] ----------")

    if result == True:
        return render_template('welcome.html', status=user_name)
    else:
        return render_template('index.html', status="Unrecognized face!")


def cleanUp(camera):
    print("Turning off camera.")
    camera.release()
    print("Camera off.")
    print("Program ended.")
    cv2.destroyAllWindows()


def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture()
    #  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
    # for local webcam use cv2.VideoCapture(0)
    key = cv2. waitKey(1)

    while True:
        try:
            # Capture frame-by-frame
            success, frame = camera.read()  # read the camera frame
            if not success or key == ord('q'):
                cleanUp(camera)
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        except(KeyboardInterrupt):
            cleanUp(camera)
            break


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camview')
def index():
    """Video streaming main page."""
    return render_template('camview.html')


def gen_frames_detection():  # generate frame by frame from camera
    # camera = cv2.VideoCapture(0)
    # Capture frame-by-frame
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("\n[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat")

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # start the video stream thread
    print("\n[INFO] starting video stream thread...")
    # vs = FileVideoStream(args["video"]).start()
    # fileStream = True
    # vs = VideoStream(src=0).start()
    # vs = VideoStream(usePiCamera=True).start() #Use for Raspberry Pi
    vs = VideoStream().start()
    fileStream = False
    time.sleep(1.0)

    j = 0
    result = False
    #  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
    # for local webcam use cv2.VideoCapture(0)

    user_name = ""
    key = cv2. waitKey(1)
    while True and result == False:
        try:
            # loop over frames from the video stream
            while True and result == False:
                try:
                    j = j+1
                    # if this is a file video stream, then we need to check if
                    # there any more frames left in the buffer to process
                    if fileStream and not vs.more():
                        break

                    # grab the frame from the threaded video file stream, resize
                    # it, and convert it to grayscale
                    # channels)
                    frame = vs.read()
                    frame = imutils.resize(frame, width=800)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # detect faces in the grayscale frame
                    rects = detector(gray, 0)
                    # loop over the face detections
                    for rect in rects:
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)

                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2.0

                        # compute the convex hull for the left and right eye, then
                        # visualize each of the eyes
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(
                            frame, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(
                            frame, [rightEyeHull], -1, (0, 255, 0), 1)

                        # check to see if the eye aspect ratio is below the blink
                        # threshold, and if so, increment the blink frame counter
                        if ear < EYE_AR_THRESH:
                            COUNTER += 1

                        # otherwise, the eye aspect ratio is not below the blink
                        # threshold
                        else:
                            # if the eyes were closed for a sufficient number of
                            # then increment the total number of blinks
                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                TOTAL += 1

                            # reset the eye frame counter
                            COUNTER = 0
                            # draw the total number of blinks on the frame along with
                        # the computed eye aspect ratio for the frame
                        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if TOTAL >= 3 or j > 101:  # 3 is how many times the face must blink
                        result = True
                        if TOTAL >= 3:
                            cv2.putText(frame, "Real face, comparing now ...", (500, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            check_extract_faces(frame)
                            decision, user_name = final_compare_blink(result)
                            cv2.putText(frame, "Welcome {}".format(user_name), (500, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        else:  # not real face, maybe photo
                            cv2.putText(frame, "Eyes blinks not detectec!", (500, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    else:
                        if j % 10 == 0:  # tictac
                            cv2.putText(frame, "Unrecognized face", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, "# Unrecognized face", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print(j)

                    # return the frame
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                except(KeyboardInterrupt):
                    vs.stop()
                    break

        except(KeyboardInterrupt):
            vs.stop()
            break

    vs.stop()
    stop_image = np.zeros((500, 500, 3), np.uint8)
    if user_name == "":
        cv2.putText(stop_image, "Unrecognized face", (150, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(stop_image, "Welcome {}".format(user_name), (150, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', stop_image)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed_detection')
def video_feed_detection():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(port=3000, debug=True)

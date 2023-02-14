# import the necessary packages
# from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import detect_blinks as db
import argparse
import glob


def mse(imageA, imageB):
    im1 = cv2.imread(imageA)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.imread(imageB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1 = cv2.resize(im1, (200, 200), interpolation=cv2.INTER_CUBIC)
    im2 = cv2.resize(im2, (200, 200), interpolation=cv2.INTER_CUBIC)
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension

    try:
        im = im2[:, :, 0]
    except:
        im = im2
    # print(im.shape)

    err = np.sum((im1.astype("float") - im.astype("float")))
    err /= float(im1.shape[0] * im2.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    print("### [MSE] : ", err, "###")
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()

# print(mse("faces/amire.jpg"))


def final_compare(user, auth="faces/auth.jpg"):
    blink = db.count_blink(min=2)

    print(" >> PREPROCESSING --------- ")
    min_mse = 1000
    most_like = ''
    for filename in glob.glob('faces/*.jpg'):
        if (filename) != auth:
            print(" - Comparing between [", auth, "] and [", filename, "]")
            val_mse = abs(mse(auth, filename))
            if val_mse < min_mse:
                min_mse = val_mse
                most_like = filename
    person_name = most_like.split(
        "/")[len(most_like.split("/"))-1].split(".")[0]
    print(" >> END PREPROCESSING --------- ")
    print("    --- the most likely person is : ",
          person_name, "[", min_mse, "]")
    result = blink and (-10.00 <= min_mse <= 10.00) and person_name == user
    print("### Result: ", result, '###')
    return result


def final_compare_blink(blink, auth="faces/auth.jpg"):
    # to use in weserver

    print(" >> PREPROCESSING --------- ")
    min_mse = 1000
    most_like = ''
    for filename in glob.glob('faces/*.jpg'):
        if (filename) != auth:
            print(" - Comparing between [", auth, "] and [", filename, "]")
            val_mse = abs(mse(auth, filename))
            if val_mse < min_mse:
                min_mse = val_mse
                most_like = filename
    person_name = most_like.split(
        "/")[len(most_like.split("/"))-1].split(".")[0]
    print(" >> END PREPROCESSING --------- ")
    print("    --- the most likely person is : ",
          person_name, "[", min_mse, "]")
    result = blink and (-10.00 <= min_mse <= 10.00)
    print("### Result: ", result, '###')
    return result, person_name


ap = argparse.ArgumentParser()
ap.add_argument("-u", "--user", type=str, required=False,
                help="User name to get his photo")
args = vars(ap.parse_args())

# print(abs(mse("faces/yakoub.jpg", "faces/yakoub.jpg")))
# final_compare(args['user'])

import cv2


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

# found here: https://stackoverflow.com/questions/61492452/how-to-check-if-opencv-is-using-gpu-or-not
def is_cuda_cv(): # 1 == using cuda, 0 = not using cuda
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            return 1
        else:
            return 0
    except:
        return 0



'''
get_orb returns key points and descriptors for the left image and optionally the right image.
if the right image is provided the function also returns the matches of the descriptors between the two images.
return: key points left image, descriptors left image, key points right image, descriptors right image, matches
'''


def get_orb(img_left, img_right=None, n_features=1000, max_matches=100, orb=None):
    if len(img_left.shape) > 2:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    if orb is None:
        orb = cv2.ORB_create(nfeatures=n_features)
    kp_left, des_left = orb.detectAndCompute(img_left, None)
    if img_right is not None:
        kp_right, des_right = orb.detectAndCompute(img_right, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_left, des_right)
        matches = sorted(matches, key=lambda x: x.distance)
        return kp_left, des_left, kp_right, des_right, matches[:max_matches]
    return kp_left, des_left, None, None, None


if __name__ == '__main__':
    # example for mono image
    img = cv2.imread("examples/16327480994671.png", 1)
    img = image_resize(img, width=600)

    kp, des, _, _, _ = get_orb(img)

    imgg = cv2.drawKeypoints(img, kp, None)

    cv2.imwrite("examples/orb.png", imgg)

    # example for stereo images
    image_left = cv2.imread("examples/image250_left.png", 1)
    image_right = cv2.imread("examples/image250_right.png", 1)
    image_left = image_resize(image_left, width=600)
    image_right = image_resize(image_right, width=600)

    lkp, ldes, rkp, rdes, matches = get_orb(image_left, image_right, max_matches=250)

    imgg = cv2.drawMatches(image_left, lkp, image_right, rkp, matches, None)

    cv2.imwrite("examples/orb_stereo.png", imgg)


    # gpu example

    if is_cuda_cv():

        cuimg = cv2.cuda_GpuMat()
        cuimg.upload(img)
        corb = cv2.cuda_ORB.create(1000)
        kp, des, _, _, _ = get_orb(cuimg, orb=corb)

        imgg = cv2.drawKeypoints(img, kp.download(), None)

        cv2.imwrite("examples/orbgpu.png", imgg)

import numpy as np

def data_augmentation(image):
    mode = np.random.randint(0,8)
    # if mode == 0:
    #     # original
    #     return image

    if mode == 1:
        # flip up and down
        image = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image,axes= (1,2))
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image,axes= (1,2))
        image = np.flip(image,axis=1)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes= (1,2))
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes= (1,2))
        image = np.flip(image,axis=1)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes= (1,2))
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=(1,2))
        image = np.flip(image,axis=1)

    # flag = np.random.randint(0, 2)
    # if flag == 1:
    #     # original
    #     image = random_exposure(image)

    # flag = np.random.randint(0, 2)
    # if flag == 1:
    #     image = random_contraction(image)

    return image

def random_exposure(image, sigma=0.01):
    '''
    use gamma function to random adjust the exposure
    note:the imput image should be normalized to (0,1)
    :param image: image
    :param sigma: a parameter to adjust the distribution that r follows
    :return:
    '''

    assert np.max(image) <= 1
    assert np.min(image) >= 0

    r = np.random.randn(1)*sigma+1

    image = image**r

    # image[image < 0] = 0
    # image[image > 1] = 1

    return image

def random_contraction(image, sigma1=0.1, sigma2=0.1):
    '''
        use gamma function to random adjust the exposure
        note:the imput image should be normalized to (0,1)
        :param image: image
        :param sigma: a parameter to adjust the distribution that slope follows
        :return: image
    '''

    assert np.max(image) <= 1
    assert np.min(image) >= 0

    a = np.random.randn(1)*sigma1+1
    b = np.random.randn(1)*sigma2

    image = a*image+b

    image[image<0] = 0
    image[image>1] = 1

    assert np.max(image) <= 1
    assert np.min(image) >= 0

    return image

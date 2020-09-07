from imutils import paths
import numpy as np
import argparse
import cv2
import os


def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
	# adding a single column (width) so we can compute the horizontal gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-d", "--dataset", required=True,
    #                 help="path to input dataset")
    # ap.add_argument("-r", "--remove", type=int, default=-1,
    #                 help="whether or not duplicates should be removed (i.e., dry run)")
    # args = vars(ap.parse_args())

    """1. Get the file paths.
    2. load the image from path and find the dhash
    3. store the hash and filepath in dict
    """
    file_paths = list(paths.list_images("C:/Users/akash/Downloads/keras tips and tricks/classify objects in video/hockey_images/"))
    dictionary = {}
    # print(file_paths)

    for file_path in file_paths:
        # print(type(file_path))
        image = cv2.imread(file_path)
        hash = dhash(image)
        # if str(hash) in dictionary.keys():
        #     dictionary[str(hash)].append(file_path)
        # else:
        #     dictionary[str(hash)] = file_path

        p = dictionary.get(hash, [])
        p.append(file_path)
        dictionary[hash] = p

    # for keys, items in dictionary.items():
        # print(type(items))

    """
    1. loop over the items of the dict and extract values which are more than 1, leaving the first item in list.
    2. append them in the new list
    3. loop over this list and delete every image from directory
    """
    new_list = []
    for hash, hash_paths in dictionary.items():
        if len(hash_paths) > 1:
            new_list.append(hash_paths[1:])

    print(new_list)
    for image_path in new_list:
        for i in image_path:
            os.remove(i)

    print("Done... deleting the duplicated images.")
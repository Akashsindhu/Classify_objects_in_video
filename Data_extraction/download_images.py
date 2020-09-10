from imutils import paths
import argparse
import requests
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True, help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True, help="path to the output directory of images")
args = vars(ap.parse_args())
print(args)

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0
print(args["output"])

for row in rows:
    try:
        image = requests.get(row, timeout=60)
        p = os.path.join(args["output"], "{}.jpg".format(total))
        f = open(p, 'wb')
        f.write(image.content)
        f.close()
        print("downloaded: {}".format(p))
        total += 1

    except:
        #handle any exceptions are thrown during the download process
        print("Error downloading {} .... skipping".format(p))

## if opencv can open the files, we keep them. Otherwise, delete them
for imagepath in paths.list_files(args["output"]):
    try:
        image = cv2.imread(imagepath)

        if image is None:
            os.remove(imagepath)

    except:
        print("Cannot open the file, so deleting it.")
        os.remove(imagepath)

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        img =  cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
        if img is not None:
            images.append(img)
    return images
    
def diff_mask(img1, img2, threshold):
    diff = cv.absdiff(img1, img2)
    mask = diff>threshold
    mask = mask*255
    return mask.astype(np.uint8)

def main():
    CWD = os.getcwd()
    folderOfSeq = 'Video sequences for project-20210918'
    seq = 'seq7'
    sequence = load_images_from_folder(os.path.join(CWD, folderOfSeq, seq))
    mean_frame = np.mean(sequence[0:10], axis=0).astype(np.uint8)
    for imagenumber in range(len(sequence)):
        # if imagenumber < 5:
        #     mean_frame = np.mean(sequence[0:5], axis=0).astype(np.uint8)
        # else:
        #     mean_frame = np.mean(sequence[imagenumber-5:imagenumber], axis=0).astype(np.uint8)
        thresholdValue = 10
        img = sequence[imagenumber]
        mask = diff_mask(mean_frame, img, thresholdValue)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel, iterations = 2)
        closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel, iterations = 4)
        contours, hierachy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Find the index of the largest contour
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        # print(cv.contourArea(cnt))
        if cv.contourArea(cnt) > 500:

            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(img,(x,y),(x+w,y+h),0,2)
        cv.imshow("closed mask", closing)
        cv.waitKey(1)
        cv.imshow("image", img)
        cv.waitKey(1)
        time.sleep(0.5)
    cv.destroyAllWindows()

    return

if __name__ == "__main__":
    main()


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


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
    return mask

def main():
    CWD = os.getcwd()
    folderOfSeq = 'Video sequences for project-20210918'
    seq = 'seq7'
    sequence = load_images_from_folder(os.path.join(CWD, folderOfSeq, seq))
    mean_frame = np.mean(sequence[0:10], axis=0).astype(np.uint8)
    plt.imshow(mean_frame, cmap="gray")
    imagenumber = 10
    thresholdValue = 10
    mask = diff_mask(mean_frame, sequence[imagenumber], thresholdValue)
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.figure()
    plt.imshow(sequence[imagenumber], cmap='gray')
    plt.show()
    
    return

if __name__ == "__main__":
    main()


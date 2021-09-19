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

def main():
    CWD = os.getcwd()
    folderOfSeq = 'Video sequences for project-20210918'
    folders = ['Seq1','Seq2','Seq3','Seq4','Seq5','Seq6','Seq7']
    for seq in folders:
        print(seq)
        sequence = load_images_from_folder(os.path.join(CWD, folderOfSeq, seq))
        for imagenumber in range(len(sequence)):
            img = sequence[imagenumber]


    return

if __name__ == "__main__":
    main()
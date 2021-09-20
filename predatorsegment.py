import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
        #cl = clahe.apply(img)
        #img = cl
    
        if img is not None:
            images.append(img)
    return images

def diff_mask(img1, img2, threshold):
    diff = cv2.absdiff(img1, img2)
    print(diff)
    mask = diff>threshold
    return mask

def main():
    sequence = load_images_from_folder("C:/Users/larki/Documents/Code/PredatorProject/Video sequences for project-20210914/Seq7")
    print(sequence)
    mean_frame = np.mean(sequence, axis=0).astype(np.uint8)
    plt.imshow(mean_frame, cmap="gray")
    print(mean_frame.shape)
    mask = (diff_mask(mean_frame, sequence[10], 20)*255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 4)


    thresh1 = cv2.adaptiveThreshold(sequence[10], 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    closing1 = cv2.morphologyEx(thresh1,cv2.MORPH_CLOSE, (3, 3), iterations = 4)
    opening1 = cv2.morphologyEx(closing1,cv2.MORPH_OPEN, kernel, iterations = 1)

    plt.figure()
    plt.imshow(opening1)
    plt.figure()
    plt.imshow(opening)
    plt.figure()
    plt.imshow(sequence[10], cmap='gray')
    plt.show()
    

if __name__ == "__main__":
    main()
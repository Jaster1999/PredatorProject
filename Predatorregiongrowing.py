import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

from numpy.lib.type_check import imag

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(img)
    return images

def diff_mask(img1, img2, threshold):
    diff = cv2.absdiff(img1, img2)
    print(diff)
    mask = diff>threshold
    return mask

def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1
    
    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))
    
    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))
    
    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    return out

def region_growing(img, seed):
    seed_points = []
    outimg = np.zeros_like(img)
    seed_points.append((seed[0], seed[1]))
    processed = []
    while(len(seed_points) > 0):
        pix = seed_points[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            print(coord)
            print(img[coord[0]][coord[1]])
            if img[coord[0]][coord[1]] != 0:
                outimg[coord[0], coord[1]] = 255
                if not coord in processed:
                    seed_points.append(coord)
                processed.append(coord)
        seed_points.pop(0)
        cv2.imshow("progress",outimg)
        cv2.waitKey(1)
    return outimg

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), img[y,x]) 
        clicks.append((y,x))

def main():
    sequence = load_images_from_folder("C:/Users/larki/Documents/Code/PredatorProject/Video sequences for project-20210914/Seq7")
    print(sequence)
    mean_frame = np.mean(sequence[0:10], axis=0).astype(np.uint8)
    plt.imshow(mean_frame, cmap="gray")
    print(mean_frame.shape)
    mask = (diff_mask(mean_frame, sequence[10], 20)*255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 4)


    thresh1 = cv2.adaptiveThreshold(sequence[10], 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    closing1 = cv2.morphologyEx(thresh1,cv2.MORPH_CLOSE, (3, 3), iterations = 4)
    opening1 = cv2.morphologyEx(closing1,cv2.MORPH_OPEN, kernel, iterations = 1)

    global img
    img = opening1
    plt.figure()
    plt.imshow(opening1)
    plt.figure()
    plt.imshow(opening)
    plt.figure()
    plt.imshow(sequence[10], cmap='gray')
    plt.show()
    global clicks
    clicks = []
    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.imshow('Input', opening1)
    cv2.waitKey()
    seed = clicks[-1]
    out = region_growing(opening1, seed)
    cv2.imshow('Region Growing', out)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
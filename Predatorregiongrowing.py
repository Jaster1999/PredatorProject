import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

from numpy.lib.type_check import imag

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
        cl = clahe.apply(img)
        img = cl
        '''hist,bins = np.histogram(img.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        img = cdf[img]'''

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
    sequence = load_images_from_folder("C:/Users/larki/Documents/Code/PredatorProject/Video sequences for project-20210914/Seq6")
    print(sequence)
    mean_frame = np.mean(sequence, axis=0).astype(np.uint8)
    plt.imshow(mean_frame, cmap="gray")
    print(mean_frame.shape)
    t_img = sequence[10]
    mask = (diff_mask(mean_frame, t_img, 20)*255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)
    eroded = cv2.erode(opening, kernel, iterations=1)

    edges = cv2.Canny(t_img, 150, 200, apertureSize = 3, L2gradient=False)

    #ret, thresh1 = cv2.threshold(t_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = cv2.adaptiveThreshold(t_img, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
    closing1 = cv2.morphologyEx(thresh1,cv2.MORPH_CLOSE, (3,3), iterations = 3)
    opening1 = cv2.morphologyEx(closing1,cv2.MORPH_OPEN, kernel, iterations = 1)
 
    tex1 = np.array([[-1, -2, -1],
                     [2, 4, 2],
                     [-1, -2, -1]])
    tex2 = np.array([[-1, 2, -1],
                     [-2, 4, -2],
                     [-1, 2, -1]])
    # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html
    # https://ww2.mathworks.cn/help/images/texture-segmentation-using-gabor-filters.html

    ret, thresh2 = cv2.threshold(cv2.absdiff(cv2.filter2D(t_img,  ddepth=-1, kernel=tex1), cv2.filter2D(t_img,  ddepth=-1, kernel=tex2)), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    global img
    img = opening1
    plt.figure("Manual texture1")
    plt.imshow(thresh2)
    plt.figure("Entropy plot")
    plt.imshow(entropy(t_img, disk(3)), cmap="gray")
    plt.figure()
    plt.imshow(edges)
    plt.figure()
    plt.imshow(opening1)
    plt.figure()
    plt.imshow(eroded)
    plt.figure()
    plt.imshow(t_img, cmap='gray')
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
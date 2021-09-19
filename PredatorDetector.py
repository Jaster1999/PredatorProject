import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# plt.ion()
# fig = plt.figure()
# subplt = fig.add_subplot(111)

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
    
def diff_mask(img1, img2, LWRthreshold, UPRthreshold):
    diff = cv.absdiff(img1, img2)
    cv.imshow("diff", diff)
    cv.waitKey(1)
    mask = cv.inRange(diff, LWRthreshold, UPRthreshold)
    return mask

def main():
    CWD = os.getcwd()
    folderOfSeq = 'Video sequences for project-20210918'
    folders = ['Seq1','Seq2','Seq3','Seq4','Seq5','Seq6','Seq7']
    for seq in folders:
        print(seq)
        path = []
        sequence = load_images_from_folder(os.path.join(CWD, folderOfSeq, seq))
        # BKground = np.mean(sequence, axis=0).astype(np.uint8)
        # BKground = np.median(sequence, axis=0).astype(np.uint8)
        BKground = np.percentile(sequence, q=75, axis=0).astype(np.uint8) #q=75% etc
        for imagenumber in range(len(sequence)):
            LWRthresholdValue = 10
            UPRthresholdValue = 255
            img = sequence[imagenumber]
            outImg = img.copy()
            outImg = cv.cvtColor(outImg, cv.COLOR_GRAY2BGR)
            mask = diff_mask(BKground, img, LWRthresholdValue, UPRthresholdValue)
            cv.imshow("mask", mask)
            # cv.imshow("image", img)
            cv.waitKey(1)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
            opening = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel, iterations = 1)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
            closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel, iterations = 2)
            contours, hierachy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # Find the index of the largest contour
            if len(contours) <1:
                pass
            else:
                areas = [cv.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt=contours[max_index]
                # print(cv.contourArea(cnt))
                if cv.contourArea(cnt) > 800:
                    x,y,w,h = cv.boundingRect(cnt)
                    cv.rectangle(outImg,(x,y),(x+w,y+h),(0,0,0),2)
                    # compute the center of the contour
                    M = cv.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv.circle(outImg, (cX, cY), 2, (0,0,0), -1)
                    location = (cX, cY)
                    path.append(location)
            if len(path) > 1:
                for point in path:
                    cv.circle(outImg, (point[0], point[1]), 3, (0,0,255), -1)
                for point in range(len(path)-1):
                    cv.line(outImg, path[point], path[point+1], (255,0,0), 2, cv.LINE_AA)
            cv.imshow("closed mask", closing)
            cv.imshow("Out Image", outImg)
            cv.waitKey(1)
            time.sleep(0.1)
        cv.destroyAllWindows()

    return

if __name__ == "__main__":
    main()


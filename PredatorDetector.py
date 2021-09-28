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
    avgPx = np.mean(img1)
    if avgPx > 80:
        # it is daytime, look for bright things
        diff = cv.subtract(img2, img1)
    else:
        #it is nighttime, look for any changes
        diff = cv.absdiff(img1, img2)
    cv.imshow("diff", diff)
    cv.waitKey(1)
    mask = cv.inRange(diff, LWRthreshold, UPRthreshold)
    return mask

def main():
    CWD = os.getcwd()
    folderOfSeq = 'Video sequences for project-20210918'
    folders = ['Seq4','Seq5']#'Seq1','Seq2','Seq3','Seq4','Seq5','Seq6','Seq7']#'Seq5']#
    for seq in folders:
        print(seq)
        areaTotal = []
        AspectTotal = []
        ConvexityTotal = []
        solidityTotal = []
        path = []
        
        animal = []
        sequence = load_images_from_folder(os.path.join(CWD, folderOfSeq, seq))
        # BKground = np.mean(sequence, axis=0).astype(np.uint8)
        # BKground = np.median(sequence, axis=0).astype(np.uint8)
        BKground = np.percentile(sequence, q=75, axis=0).astype(np.uint8) #q=75% etc
        for imagenumber in range(len(sequence)):
            LWRthresholdValue = 8
            UPRthresholdValue = 255
            img = sequence[imagenumber]
            outImg = img.copy()
            outImg = cv.cvtColor(outImg, cv.COLOR_GRAY2BGR)
            mask = diff_mask(BKground, img, LWRthresholdValue, UPRthresholdValue)
            # cv.imshow("mask", mask)
            # cv.imshow("image", img)
            cv.waitKey(1)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
            closed = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel, iterations = 1)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
            opening = cv.morphologyEx(closed,cv.MORPH_OPEN,kernel, iterations = 1)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(40,40))
            closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel, iterations = 1)

            contours, hierachy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # Find the index of the largest contour
            if len(contours) <1:
                pass
            else:
                areas = [cv.contourArea(c) for c in contours]
                
                max_index = np.argmax(areas)
                cnt=contours[max_index]
                # print(cv.contourArea(cnt))
                if cv.contourArea(cnt) > 700 and cv.contourArea(cnt) < 3000: #only animal contours should remain
                    # Use a min area bounding box to help find height and width of animal
                    rect = cv.minAreaRect(cnt)
                    box=cv.boxPoints(rect)
                    lengths = []
                    for point in range(len(box)):
                        if point ==3:
                            nextPoint = 0
                        else:
                            nextPoint = point+1
                        x = box[point][0]-box[nextPoint][0]
                        y = box[point][1]-box[nextPoint][1]
                        dist = np.sqrt(x**2 + y**2)
                        lengths.append(dist)
                    lengths.sort()
                    shortSide = lengths[0]
                    longSide = lengths[2]
                    #use the height and width to find aspect ratio
                    aspectRatio = longSide/shortSide

                    # compute the center of the contour
                    M = cv.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    location = (cX, cY)
                    path.append(location)

                    #Finding features of the contours
                    areaTotal.append(cv.contourArea(cnt))
                    perimeter = cv.arcLength(cnt, True)
                    CntConvHull = cv.convexHull(cnt)
                    ConvHullPerim = cv.arcLength(CntConvHull, True)
                    Convexity = ConvHullPerim/perimeter
                    ConvexityTotal.append(Convexity)
                    AspectTotal.append(aspectRatio)
                    convexArea=cv.contourArea(CntConvHull)
                    solidity = cv.contourArea(cnt)/convexArea
                    solidityTotal.append(solidity)

                    #Stoat = 1, Rat = 2, Hedgehog = 3
                    if aspectRatio < 1.5:
                        animal.append(3)
                    elif aspectRatio >=1.5 and aspectRatio <2.8:
                        #it is hard to distingush rat and stoat based on aspect ratio
                        animal.append(2)
                    elif aspectRatio >=2.8:
                        # greater than this aspect is definitely a stoat
                        animal.append(1)
                    
            if len(path) > 1:
                for point in path:
                    cv.circle(outImg, (point[0], point[1]), 3, (0,0,255), -1)
                for point in range(len(path)-1):
                    cv.line(outImg, path[point], path[point+1], (255,0,0), 2, cv.LINE_AA)
            cv.imshow("closed mask", closing)
            cv.imshow("Out Image", outImg)
            cv.waitKey(1)
            time.sleep(0.1)
            # cv.waitKey()
        if len(path) < 10:
            #animal not sufficiently detected
            print("animal not sufficiently detected")
            animal = [0]
        animalID = round(np.mean(animal))
        print(f"Median area: {np.median(areaTotal)}")
        print(f"Median Convexity: {np.median(ConvexityTotal)}")
        print(f"Median Aspect Ratio: {np.median(AspectTotal)}")
        print(f"Median Solidity: {np.median(solidityTotal)}")
        print(f"Animal in Seq is: {animalID}")
        if animalID == 0:
            text = "No Animal"
        elif animalID == 1:
            text = "Stoat"
        elif animalID == 2:
            text = "Rat"
        elif animalID == 3:
            text = "Hedgehog"
        else:
            text = "Unknown"
        outImg = cv.putText(outImg, "Animal Found: "+text, (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
        cv.imshow("Out Image", outImg)
        cv.waitKey()
        cv.destroyAllWindows()

    return

if __name__ == "__main__":
    main()


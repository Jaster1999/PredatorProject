import cv2 as cv
import numpy as np
import os
import time

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder): #find all the images in the folder and save them to a list
        img = cv.imread(os.path.join(folder,filename))
        img =  cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Make image grayscale
        if img is not None:
            images.append(img)
    return images
    
def diff_mask(img1, img2, LWRthreshold, UPRthreshold):
    avgPx = np.mean(img1) #measure the average value of the pixels in the image
    if avgPx > 80:
        # it is daytime, look for bright things
        diff = cv.subtract(img2, img1)
    else:
        #it is nighttime, look for any changes
        diff = cv.absdiff(img1, img2)
    cv.imshow("diff", diff)
    cv.waitKey(1)
    mask = cv.inRange(diff, LWRthreshold, UPRthreshold)
    #make a mask of the diff the is between an upper and lower threshold
    return mask

def main():
    CWD = os.getcwd()
    folderOfSeq = 'Video sequences for project-20210918'
    folders = ['Seq1','Seq2','Seq3','Seq4','Seq5','Seq6','Seq7']
    #Run through each Seq
    for seq in folders:
        print(seq)
        areaTotal = []
        AspectTotal = []
        ConvexityTotal = []
        solidityTotal = []
        path = []
        animal = []
        sequence = load_images_from_folder(os.path.join(CWD, folderOfSeq, seq))
        #read all the images of in the seq
        BKground = np.percentile(sequence, q=75, axis=0).astype(np.uint8) #q=75% etc
        #Create a background image from the 75 percentile of the seq
        for imagenumber in range(len(sequence)): # loop through all images in the sequence
            LWRthresholdValue = 8
            UPRthresholdValue = 255
            img = sequence[imagenumber]
            outImg = img.copy()
            #make a copy of the input image do modify, ie draw lines on.
            outImg = cv.cvtColor(outImg, cv.COLOR_GRAY2BGR)
            mask = diff_mask(BKground, img, LWRthresholdValue, UPRthresholdValue)
            #Get the mask of what could be animals
            cv.imshow("mask", mask)
            cv.waitKey(1)
            #perform Morphological opening and closing to remove noise and join the chunks of "animal"
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
            closed = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel, iterations = 1)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
            opening = cv.morphologyEx(closed,cv.MORPH_OPEN,kernel, iterations = 1)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(40,40))
            closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel, iterations = 1)

            #Closing image should now contain big blobs for animal
            contours, hierachy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            #Find all the blobs in the image
            
            if len(contours) <1:
                pass    # If there are no Blobs in the img, there is no animal.
            # Find the index of the largest contour
            else:
                areas = [cv.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt=contours[max_index]

                if cv.contourArea(cnt) > 700 and cv.contourArea(cnt) < 3000: #only animal contours should remain
                    # Use a min area bounding box to help find height and width of animal
                    rect = cv.minAreaRect(cnt)
                    box=cv.boxPoints(rect)
                    lengths = []
                    #Box is a list of points that make up the corners of the min area rect
                    for point in range(len(box)):
                        #computing the lengths of the sides of the box
                        if point ==3:
                            nextPoint = 0
                        else:
                            nextPoint = point+1
                        x = box[point][0]-box[nextPoint][0]
                        y = box[point][1]-box[nextPoint][1]
                        dist = np.sqrt(x**2 + y**2)
                        lengths.append(dist)
                    #sorting the lengths list into [short, short, long, long]
                    lengths.sort()
                    shortSide = lengths[0]
                    longSide = lengths[2]
                    #use the height and width to find aspect ratio
                    aspectRatio = longSide/shortSide

                    # compute the center of the contour
                    M = cv.moments(cnt)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    location = (cX, cY) #coordinate of the Centre Of Mass (COM) of the contour
                    #Append the location of the COM of the Blob, to a list of points that track where the animal went.
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
                        animal.append(2)
                    elif aspectRatio >=2.8:
                        # greater than this aspect is definitely a stoat
                        animal.append(1)
            #draw the path on the out image        
            if len(path) > 1:
                for point in path:
                    cv.circle(outImg, (point[0], point[1]), 3, (0,0,255), -1)
                for point in range(len(path)-1):
                    cv.line(outImg, path[point], path[point+1], (255,0,0), 2, cv.LINE_AA)
            cv.imshow("closed mask", closing)
            cv.imshow("Out Image", outImg)
            cv.waitKey(1)
            time.sleep(0.05)
            # cv.waitKey()
        #after looping through every image in the sequence, Check to see if an animal was successfully 
        if len(path) < 5:
            #animal not sufficiently detected
            print("animal not sufficiently detected")
            animal = [0]
        #take an average of what it thinks the animal is, round to int
        animalID = round(np.mean(animal))
        # These lines were for displaying the features of the contours
        # print(f"Median area: {np.median(areaTotal)}")
        # print(f"Median Convexity: {np.median(ConvexityTotal)}")
        # print(f"Median Aspect Ratio: {np.median(AspectTotal)}")
        # print(f"Median Solidity: {np.median(solidityTotal)}")
        
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
        print("Animal in"+ seq + "is: " + text)
        #put some information on the output image
        outImg = cv.putText(outImg, "Animal Found: "+text, (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
        cv.imshow("Out Image", outImg)
        cv.waitKey() #wait for user to press key to move to next seq
        cv.destroyAllWindows()

    return

if __name__ == "__main__":
    main()


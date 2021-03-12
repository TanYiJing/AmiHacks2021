"""
Demonstration of the GazeTracking library.2
Check the README.md for complete documentation.
"""

import collections 
import argparse
import numpy as np
import cv2
import os
import re
from matplotlib import pyplot as plt 
from gaze_tracking import GazeTracking
from tqdm import tqdm

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
#width
webcam.set(3, 640)
#height
webcam.set(4, 480)
def collect_gaze_data() :   
    parser=argparse.ArgumentParser()
    parser.add_argument('--image')
    
    args=parser.parse_args()
    
    openCvFaceModel="age_gender_model/opencv_face_detector.pbtxt"
    openCvFaceModelDetector="age_gender_model/opencv_face_detector_uint8.pb"
    ageProto="age_gender_model/age_deploy.prototxt"
    ageClassificationCNN="age_gender_model/age_net.caffemodel"
    genderProto="age_gender_model/gender_deploy.prototxt"
    genderClassificationCNN="age_gender_model/gender_net.caffemodel"
    
    #this mean values get from internet as they say this would yield best result for the model
    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-3)', '(4-7)', '(8-14)', '(15-20)', '(21-36)', '(38-46)', '(48-58)', '(60-100)']
    genderList=['Male','Female']
    
    #deepneuralnetwork for accessing pretrain data
    faceCNN=cv2.dnn.readNet(openCvFaceModelDetector,openCvFaceModel)
    ageCNN=cv2.dnn.readNet(ageClassificationCNN,ageProto)
    genderCNN=cv2.dnn.readNet(genderClassificationCNN,genderProto)
    
    
    padding=20
    
    while True:
       
        hasFrame,frame=webcam.read()
        if not hasFrame:
            cv2.waitKey()
            break
        image = cv2.imread("sample.png") 
        cv2.imshow("test", image)
        #detect whether the frame got face or not by providing the faceCNN model and the video frame
        resultImg,faceBoxes=highlightFace(faceCNN,frame)
        #no face detected just continue to stream the frame
        if not faceBoxes:
            cv2.putText(frame, "Left Pupil Gaze Coordinate :  None", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Right Pupil Gaze Coordinate : None" , (90, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Age : None" , (90, 120),  cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame,"Gender : None" , (90, 150),  cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.imshow("FYP", frame)
        #face detected    
        for faceBox in faceBoxes:
            
            #crop out only the face in the whole frame 
            face=frame[max(0,faceBox[1]-padding):
                       min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                       :min(faceBox[2]+padding, frame.shape[1]-1)]
            
            cropped_face=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            #identify user gender by using the the genderCNN model by passing in the cropped face
            genderCNN.setInput(cropped_face)
            genderPreds=genderCNN.forward()
            gender=genderList[genderPreds[0].argmax()]
            
            #throw in the cropped face into the ageCNN model to evaluate the age 
            ageCNN.setInput(cropped_face)
            agePreds=ageCNN.forward()
            age=ageList[agePreds[0].argmax()]
            
            #throw in the whole video frame to identify the gaze coordinates
            gaze.refresh(frame)
            #framepupil = gaze.annotated_frame()
            
            #return the coordinates of the left pupil
            coordinate_left_pupil = gaze.pupil_left_coords()
            #return the coordinates of the left pupil
            coordinate_right_pupil = gaze.pupil_right_coords()
            
            cv2.putText(frame, "Left Pupil Gaze Coordinate :  " + str(coordinate_left_pupil), (90, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Right Pupil Gaze Coordinate : " + str(coordinate_right_pupil), (90, 90), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame, "Age : "+ f'{age}', (90, 120),  cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.putText(frame,"Gender : " + f'{gender}', (90, 150),  cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            cv2.imshow("FYP", frame)
            
            
            
            if os.path.exists("gaze_data.txt"):
                file_mode = 'a' # append if already exists
            else:
                file_mode = 'w' # make a new file if not
            
            
            if str(coordinate_left_pupil) != "None" and str(coordinate_right_pupil) != "None" :
                left = re.sub('[(,)]', '', str(coordinate_left_pupil)).split() 
                right = re.sub('[(,)]', '', str(coordinate_right_pupil)).split() 
                left_coordinate = int((int(left[0]) + int(right[0])) /2)
                right_coordinate = int(int(left[1]) + int(right[1]) /2)
                #print("l:"+str(left))
                #print("r:"+str(right))
                #print("la:"+str(left_coordinate))
                #print("ra:"+str(right_coordinate))
                f = open("gaze_data.txt", file_mode)
                f.write(str(left_coordinate)+","+str(right_coordinate)+"\n")
                f.close()
            
            if os.path.exists("potential_customer.txt"):
                file_mode = 'a' # append if already exists
            else:
                file_mode = 'w' # make a new file if not
                
            if f'{gender}' != '' and f'{age}' != '':
                f = open("potential_customer.txt", file_mode)
                f.write(str(f'{gender}')+","+str(f'{age}')+"\n")
                f.close()
                
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows() 
    cv2.VideoCapture(0).release()
           
        
    
        
        

def GaussianMask(sizex,sizey, sigma=33, center=None,fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0])==False and np.isnan(center[1])==False:            
            x0 = center[0]
            y0 = center[1]        
        else:
            return np.zeros((sizey,sizex))

    return fix*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def heatMapPlotter(GazeCoordinates, width, height, imgfile, alpha=0.5, threshold=10):
    """
    GazeCoordinates   : gaze coordinate x,y and fixation in numpy format
    width     : output heatmap img width
    height    : output heatmap img height
    imgfile   : input image pass to plot heatmap and will generate an output image
    alpha     : marge rate imgfile and heatmap 
    threshold : heatmap threshold(0~255)
    """
    
    heatmap = np.zeros((height,width), np.float32)
    for n_subject in tqdm(range(GazeCoordinates.shape[0])):
        heatmap += GaussianMask(width, height, 33, (GazeCoordinates[n_subject,0],GazeCoordinates[n_subject,1]),
                                100)

    # Normalization
    heatmap = heatmap/np.amax(heatmap)
    heatmap = heatmap*255
    heatmap = heatmap.astype("uint8")
    
    if imgfile.any():
        # Resize heatmap to imgfile shape 
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create mask
        mask = np.where(heatmap<=threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
        marge = imgfile*mask + heatmap_color*(1-mask)
        marge = marge.astype("uint8")
        marge = cv2.addWeighted(imgfile, 1-alpha, marge,alpha,0)
        return marge

    else:
        # applying color heatmap 
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) 
        return heatmap
    
def plotHeatMap ():
    #open text file and read the coordinates
    f = open('gaze_data.txt') 
    line = f.readline()
    # remove all symbol 
    x=line.split(",")
    
    #print(x[0])
    #first line of coordinate is converted into numpy array
    GazeCoordinates = np.array([[int(x[0]),int(x[1]),1]])  
    while line:
        line = f.readline()
        if line != '':
             # remove all symbol 
             x=line.split(",")
             # append data from text file into numpy array 
             GazeCoordinates=np.append(GazeCoordinates, [[int(x[0]),int(x[1]),1]],axis = 0)       
    f.close()
    #input ur image, name the file accordingly 
    img = cv2.imread('sample.png') 
    height, width, _ = img.shape
    heatmap = heatMapPlotter(GazeCoordinates, width, height, img, 0.7, 5)
    #create a file called heatMapResult.png
    cv2.imwrite("heatMapResult.png",heatmap)
    

def highlightFace(faceCNNModel, frame, conf_threshold=0.7):
    #copy the frame data into a variable 
    faceFrame=frame.copy()
    #get how big is the frame
    frameHeight=faceFrame.shape[0]
    frameWidth=faceFrame.shape[1]
    #detect whether face is detected ot not using the faceCNN passed in 
    blob=cv2.dnn.blobFromImage(faceFrame, 1.0, (300, 300), [104, 117, 123], True, False)
    
    faceCNNModel.setInput(blob)
    faces_detected=faceCNNModel.forward()
    faceBoxes=[]
    for i in range(faces_detected.shape[2]):
        confidence=faces_detected[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(faces_detected[0,0,i,3]*frameWidth)
            y1=int(faces_detected[0,0,i,4]*frameHeight)
            x2=int(faces_detected[0,0,i,5]*frameWidth)
            y2=int(faces_detected[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(faceFrame, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return faceFrame,faceBoxes

def generatePotentialCustomerAgeBase():
    customer_gender = []
    customer_age=[]
    ageList =[]
    percent=[]
    f = open('potential_customer.txt')
    
    line = f.readline()
    
    while line:
        x = line.split(',')
        customer_gender.append(x[0])
        customer_age.append(x[1])
        line = f.readline()
        
    f.close()    
    freq = collections.Counter(customer_age)
    for (key, value) in freq.items():
        percentage =  (value/len(customer_age))*100
        #print (key, " -> ", "{:.2f}".format(percentage)) 
        ageList.append(key)
        percent.append("{:.2f}".format(percentage))
        
    
  
    plt.rcParams['font.size'] = 18
    fig = plt.figure(figsize =(8, 8)) 
    plt.pie(percent, labels = ageList,autopct='%1.2f%%') 
    

    plt.figtext(.5,.9,'Potential Customer Base', fontsize=30, ha='center')
    plt.figtext(.5,.86,'By Age',fontsize=30,ha='center')
    plt.show()

def generatePotentialCustomerAgeGender():
    customer_gender = []
    genderList =[]
    percent=[]
    f = open('potential_customer.txt')
    
    line = f.readline()
    
    while line:
        x = line.split(',')
        customer_gender.append(x[0])
       
        line = f.readline()
        
    f.close()    
    freq = collections.Counter(customer_gender)
    for (key, value) in freq.items():
        percentage =  (value/len(customer_gender))*100
        #print (key, " -> ", "{:.2f}".format(percentage)) 
        genderList.append(key)
        percent.append("{:.2f}".format(percentage))
        
    
  
    plt.rcParams['font.size'] = 18
    fig = plt.figure(figsize =(8, 8)) 
    plt.pie(percent, labels = genderList,autopct='%1.2f%%') 
    

    plt.figtext(.5,.9,'Potential Customer Base', fontsize=30, ha='center')
    plt.figtext(.5,.86,'By Gender',fontsize=30,ha='center')
    plt.show()
    
if __name__ == '__main__':
   print("Welcome To Our FYP ")
   print("1. Gather Data")
   print("2. Generate Heatmap")
   print("3. Generate Potential Customer Base Age")
   print("4. Generate Potential Customer Base Gender")
   print("Please input a selection : ")
   selection =int(input())
   while str(selection) != "-1":
       
   
       if selection == 1 :
           collect_gaze_data()
         
       elif selection == 2 :
            plotHeatMap()
        
       elif selection == 3 :
            generatePotentialCustomerAgeBase()
            
       elif selection == 4 :
            generatePotentialCustomerAgeGender()
       print("Welcome To Our FYP ")
       print("1. Gather Data")
       print("2. Generate Heatmap")
       print("3. Generate Potential Customer Base Age")
       print("4. Generate Potential Customer Base Gender")
       print("Please input a selection : ")
       selection =int(input())
  
   
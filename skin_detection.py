import cv2
from mtcnn.mtcnn import MTCNN
import time
prev_frame_time=0
import numpy as np
detector = MTCNN()
cap = cv2.VideoCapture(0)
k=0
num=20
import os
direc=os.getcwd()
while True:
    #k+=1
    #Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
  
    # Calculating the fps
  
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
  
    # converting the fps into integer
    fps = int(fps)
  
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
  
    # puting the FPS count on the frame
      
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
 
            frame=np.array(frame[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]])
            if frame.size != 0:
                RGBA_image = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
                HSV_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                YCbCr_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                binary_mask_image = HSV_image
                #rgba mask
                lower_values = np.array((20, 40, 95, 15), dtype = "uint8")
                upper_values = np.array([255, 255, 255, 255], dtype = "uint8")
                mask_rgba = cv2.inRange(RGBA_image, lower_values, upper_values)
                mask_rgba[(RGBA_image[:,:,2] > RGBA_image[:,:,0]) & (RGBA_image[:,:,2] > RGBA_image[:,:,1]) & (np.abs((RGBA_image[:,:,2] - RGBA_image[:,:,1])) > 15)] = 1
                #hsv mask
                lower_HSV_values = np.array([0, 10, 60], dtype = "uint8")
                upper_HSV_values = np.array([17, 150, 255], dtype = "uint8")
                #ycbcr mask
                lower_YCbCr_values = np.array((80, 105, 140), dtype = "uint8")
                upper_YCbCr_values = np.array((235, 135, 165), dtype = "uint8")
                mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
                mask_YCbCr[(YCbCr_image[:,:,2] <= (YCbCr_image[:,:,1]*1.5862 + 20)) & (YCbCr_image[:,:,2] >= (YCbCr_image[:,:,1]*0.3448 + 76.2069)) & (YCbCr_image[:,:,2] >= (YCbCr_image[:,:,1]*-4.5652 + 234.5652)) & (YCbCr_image[:,:,2] <= (YCbCr_image[:,:,1]*-1.15 + 301.75))  & (YCbCr_image[:,:,2] <= (YCbCr_image[:,:,1]*-2.2857 + 432.85))]=1
                mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values) 
                binary_mask_image = cv2.add(mask_HSV,mask_YCbCr)
                binary_mask_image = cv2.add(binary_mask_image,mask_rgba)
                #morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                image_foreground = cv2.erode(binary_mask_image,kernel, iterations = 3)  #remove noise
                dilated_binary_image = cv2.dilate(binary_mask_image,kernel,iterations = 3)   #The background region is reduced a little because of the dilate operation
                ret,image_background = cv2.threshold(dilated_binary_image,1,128,cv2.THRESH_BINARY)  #set all background regions to 128
                #watershed algortithim
                image_marker = cv2.add(image_foreground,image_background)   #add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
                image_marker32 = np.int32(image_marker) #convert to 32SC1 format
                cv2.watershed(frame,image_marker32)
                m = cv2.convertScaleAbs(image_marker32) #convert back to uint8 

                #bitwise of the mask with the input image
                ret,image_mask = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                output = cv2.bitwise_and(frame,frame,mask = image_mask)
            else:
                output=np.zeros((100,100))
                frame=output  
        
    else:
        continue
    #display resulting frame\
    #Verti = np.concatenate((output, frame), axis=1)
    cv2.putText(output, fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('skin',output)
    cv2.imshow('face',frame)
    
    #save=f'{direc}/image_{k}.jpg'
    #cv2.imwrite(save,frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
#Images are stored int directory where the notebook is present
cap.release()
cv2.destroyAllWindows()
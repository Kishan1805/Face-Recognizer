import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #absolute path
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.2,minNeighbors=5)
    return faces,gray_img

	
def labels_for_training_data(directory):
    faces=[]
    faceID=[]
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue

            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img_path:",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
               continue
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer


def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)


def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)


faces,faceID=labels_for_training_data('trainingImages') #absolute path of directory
face_recognizer=train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')


name={0:"Priyanka",1:"akshay"}

# this is for single image
test_img=cv2.imread('testingimage.jpg')   #absolute path of directory
faces_detected,gray_img=faceDetection(test_img)
print("faces_detected:",faces_detected)


for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    roi_gray = cv2.resize(roi_gray,(100,100))
    label,confidence=face_recognizer.predict(roi_gray)
    print("confidence:",confidence)
    print("label:",label)
    draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>100):
        continue
    put_text(test_img,predicted_name,x,y)


resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

# this for finding accuracy for testingimage folder
def get_accuracy(directory):
    total=0
    accurate=0
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")
                continue

            id=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img_path:",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect,gray_img=faceDetection(test_img)
            if len(faces_rect)!=1:
               continue
            total=total+1;
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]
            roi_gray = cv2.resize(roi_gray,(100,100))
            label,confidence=face_recognizer.predict(roi_gray)
            print(label)
            print("confidence:",confidence)
            if(confidence>100):
                continue
            if(int(id)==label):
                accurate=accurate+1
    print(total)
    print(accurate)
    print(float(accurate/total))


# In[31]:


get_accuracy('testingImages')  #absolute path of directory


# In[ ]:





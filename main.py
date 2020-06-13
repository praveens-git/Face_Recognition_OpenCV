import numpy as np
import cv2
from PIL import Image
import os


print("Facial Recognition using OpenCV")

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image_source = 'Images/'
user_path = 'Data/user_list.txt'
trained_data = "Data/reference.yml"
id = []


def check_user(dirname):
    global name
    if (dirname in os.listdir(image_source)):
        print("Name already Exists.\nDo you want to overwrite (y/n): ",end="")
        if (input() == 'n'):
            name = input("Enter Your Name : ")
            check()

def user_data(operation, new=""):
    global user_list
    if (operation == 'w'):
        user_file = open(user_path, 'a+')
        user_file.write(new + "\n")
        user_file.close()


    elif (operation == 'r'):
        user_file = open(user_path, 'r')
        user_list = user_file.read()
        user_list = user_list.split("\n")
        user_file.close()

user_data('r')

def Train():
    user_data('r')
    print("\nYour Image is being Processed")
    print("Please wait...........")
    sampling=[]
    id=[]
    for dir in os.listdir(image_source):
        for i in range(10):
            face = np.array(Image.open(image_source+dir+"/"+str(i+1)+".jpg").convert('L') ,  'uint8')
            index = user_list.index(dir)
            output = face_cascade.detectMultiScale(face)

            for x, y, w, h in output:
                sampling.append(face[y:y+h,x:x+w])
                id.append(index)

    recognizer.train(sampling,np.array(id))
    recognizer.save(trained_data)
    print("\nTraining Complete.")
    print("Total Person Trained is {}".format(len(os.listdir(image_source))))

def capture():
    camera = cv2.VideoCapture(0)
    name = input("Enter Your Name : ")
    check_user(name)
    os.system("mkdir Images/"+name)
    user_data('w',name)
    no = 1
    check, frame = camera.read()
    cv2.imwrite("Images/" + name + "/Sample.jpg", frame)

    while(1):
        check,frame = camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        output = face_cascade.detectMultiScale(gray,1.3,5)

        for x,y,w,h in output:
            cv2.imwrite("Images/" + name + "/" + str(no) + ".jpg", gray[y:y+h,x:x+w])
            no += 1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)

        cv2.imshow("Video Feed", frame)

        k = cv2.waitKey(100) & 0xff
        if (k == 27) or (no > 10):
            camera.release()
            cv2.destroyAllWindows()
            Train()
            break

def Recognize():
    camera = cv2.VideoCapture(0)
    recognizer.read(trained_data)
    while (1):
        check, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in output:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

            if(confidence<100):
                name = user_list[id]
            elif(confidence>100):
                name = user_list[0]
            cv2.putText(frame, str(name), (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv2.putText(frame, str("{:.2f}".format(100 - confidence))+ '%', (x+5,y+h-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1)


        cv2.imshow("Video Feed", frame)

        k = cv2.waitKey(100) & 0xff
        if (k == 27):
            camera.release()
            cv2.destroyAllWindows()
            break

while(1):
    option = input("\n1. Train\n2. Recognize\n3. Press 'e' for exit.\nEnter your option: ")
    if(option == '1'):
        user_data('r')
        capture()
    if(option=='2'):
        user_data('r')
        Recognize()
    if(option == 'e' or option == '3'):
        print("\nQuitting......")
        break

cv2.destroyAllWindows()

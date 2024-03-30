
from ultralytics import YOLO
import cv2
import cvzone
import math
import os
from collections import Counter
#from memory_profiler import profile
import time

#@profile

def options_YOLO(choices):
    model = YOLO('training6/best.pt')
    classNames = ["cat","dog","drone", "phone"]

    if choices == 1:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            success, img  = cap.read()
            #img = cv2.flip(img, 1)
            results = model(img, stream=True)

            
            for i in results:
                boxes = i.boxes
                
                for box in boxes:
                    #confidence
                    conf = math.ceil((box.conf[0]*100))/100
                    
                    if conf > 0.40:       
                        #Bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                        
                        w, h = x2-x1, y2-y1
                        bbox = (x1, y1, w, h)
                        cvzone.cornerRect(img, bbox)
                        
                        
                        #print(conf)
                        
                        #class name
                        cls = int(box.cls[0])
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0,x1), max(40,y1)))
                        #print(classNames[cls])
            
            
            cv2.imshow("Image", img)

            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()


    elif choices == 2:
        det_res = []
        choices = ["cat", "dog", "drone", "phone"]
        #choices = ["ok"]
        init_num = 0
        #get audio file time length
        directory = "trytovalidate"
        files = os.listdir(directory)
        num_cat = len(choices)
        start_time = time.time()
        
        for i in range(num_cat):
            #cat = next(os.walk(directory))[1][i]
            cat = choices[i]
            print(cat)
        
            ans = os.listdir(str(directory + '/' + cat))
            files2 = len(ans)
            
        
            for j in range(files2):
                directory2 = "trytovalidate/{}".format(cat)
                cat2 = next(os.walk(directory2))[2][j]
                audio_path  = os.listdir(str(directory+ '/' + cat))
                path_name = os.path.join('trytovalidate', cat, audio_path[j])
                init_num += 1
                results = model(path_name)
                
                for i in results:
                    boxes = i.boxes

                    if (len(i.boxes.cls)) > 0:
                        for box in boxes:
                            conf = math.ceil((box.conf[0]*100))/100

                            if conf > 0.60:
                                cls = int(box.cls[0])
                                classN = classNames[cls]
                                det_res.append(f'{cat}: {classN}')

                    else:
                        det_res.append(f'{cat}: ERROR')
        print(Counter(det_res))
        end_time = time.time()
        print(f' Tested done with: {init_num} dataset')
        elapsed_time = end_time - start_time
        print(f' Simulaiton done within: {elapsed_time} time (s)')


    elif choices == 3:
        det_res = []
        #fileN = 'trytovalidate/cat/2023-02-22T161224Z_1016035325_RC2RFZ9T3RH1_RTRMADP_3_MEXICO-PRISON-CAT.JPG.jpg'
        #fileN = 'trytovalidate/phone/xiaomi-12-review-2.jpg'
        fileN = 'trytovalidate/phone/image12.jpeg'
        results = model(fileN)
        for i in results:
            boxes = i.boxes
            print (math.ceil((i.boxes.conf[0]*100))/100)
            if (len(i.boxes.cls)) > 0:
                for box in boxes:
                    conf = math.ceil((box.conf[0]*100))/100
                    print(conf)
                    if conf > 0.60:
                        cls = int(box.cls[0])
                        classN = classNames[cls]
                        det_res.append(classN)
                        yyy = 'True'
                        
            else:
                print('GG!')
                yyy = 'False'

            print(det_res)

    elif choices == 4:
        print("Yes!")
            
    else:
        pass



if __name__ == "__main__":
    choices = 1
    options_YOLO(choices)

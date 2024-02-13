import cv2
def draw_box(img,point_and_size): 
    for x,y,w,h in point_and_size : 
        print(w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("result",img)
    cv2.waitKey(0)

def main():
    img =  cv2.imread('./people1.jpg')
    #img = cv2.resize(img,(800,600))
    print(img.shape)
    ## ! CascadeClassifier 是 Size-sensitive,輸入的圖片大小不同會造成結果的不同 
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier("../Cascades/haarcascade_eye.xml")
    
    detector = face_detector.detectMultiScale(img_gray,scaleFactor=1.16,minNeighbors=5,maxSize=(40,40))
    
    draw_box(img,detector)
    print(img_gray.shape)
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
main()


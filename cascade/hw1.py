import cv2

def draw_box(img,point) :
    for x,y,w,h in point : 
        print(w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.imshow("test",img)
    cv2.waitKey(0)
def main():
    img = cv2.imread('./img/car.jpg')
    cv2.imshow('test',img)
    img = cv2.resize(img,(702,429))
    print(img.shape)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    car_detction = cv2.CascadeClassifier("./xml/cars.xml")
    
    detction = car_detction.detectMultiScale(img_gray,scaleFactor=1.07,minNeighbors=1,minSize=(30,30),maxSize=(60,60))
    draw_box(img,detction)
    
main()
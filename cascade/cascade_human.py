import cv2

def draw_box(img,point_and_size1,point_and_size2): 
    for x,y,w,h in point_and_size1 : 
        print(w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
    for x,y,w,h in point_and_size2 : 
        print(w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)    
    cv2.imshow("result",img)
    cv2.waitKey(0)
    
def eye_classifier(img):
    eye_detector = cv2.CascadeClassifier("./xml/haarcascade_eye.xml")
    
    detector = eye_detector.detectMultiScale(img,scaleFactor=1.16,minNeighbors=5,maxSize=(40,40))
    
    print(img.shape)
    return detector  

def face_classifier(img):
    face_detector = cv2.CascadeClassifier("./xml/haarcascade_frontalface_default.xml")
    print(img.shape)
    detections = face_detector.detectMultiScale(img,scaleFactor=1.3,minSize=(30,30))
    # * Haarcascade parameters :
    # * scaleFactor ( default = 1.1  ) : 
    # *                           對於分析的細度(越小則越精 -> 結果較少, 執行時間++)
    # *                                        (越多則越粗 -> 結果較多, 執行時間--)
    # * 以此例來說，(a) 因圖中最小的臉已被檢測出，但卻檢出預期外的錯誤，故可往精細度較低方面調整 -> scaleFactor = 1.2
    # * 以此例來說，(b) 除卻檢出預期外的錯誤，故可往精細度較高方面調整 -> scaleFactor = 1.09
    
    # * minNeighbors ( default = 3  ) : 被認為成目標，所需的被檢測出的次數
    # *                                 若為目標，基本上被檢測出的次數(Neighbor)會有一定的數量，可以用此參數來濾掉意料外的物件
    # *                                 可以設成 0 來看全物件的檢測狀況
    # * minSize (default = (30,30) )和 maxSize 設置檢測對象的最大最小值，低於 minSize 和高於 maxSize 的話就不會檢測出來。
         
    print(f'detections : \n {detections}   \n len : {len(detections)}')
     
    return detections  
def main ():
    img =  cv2.imread('./img/people1.jpg')
    print(img.shape)
    ## ! CascadeClassifier 是 Size-sensitive,輸入的圖片大小不同會造成結果的不同 
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    result1 = face_classifier(img_gray)
    result2 = eye_classifier(img_gray)
    
    draw_box(img,result1,result2)                     
main()
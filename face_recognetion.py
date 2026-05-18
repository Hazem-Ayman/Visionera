import cv2 as cv
import mediapipe as mp 


mp_faceDetection = mp.solutions.face_detection
ret = True

def process_image(img , face_detection) : 
    H,W,_ = img.shape
    
    img_rgb = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    
    out = face_detection.process(img_rgb)
    
    # print(out)
    
    if out.detections is not None : 
        for i in out.detections : 
            
            loc_data = i.location_data
            
            bbox = loc_data.relative_bounding_box
            
            x1,y1,w,h = bbox.xmin ,bbox.ymin , bbox.width ,bbox.height
            
            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)
            
            img = cv.rectangle(img , (x1,y1) , (x1+w , y1+h) ,(0,255,0) ,3 )     
            
    return img       
    

with mp_faceDetection.FaceDetection(model_selection = 0 , min_detection_confidence = 0.5) as face_detection: 
    
    webcam = cv.VideoCapture(0)
    
    
    
    
    
    # x = cv.imread( r"C:\Users\DELL\Downloads\81RpptIh-VS-e1730357104322-1200x772.jpg")
    # y = process_image(x ,face_detection)
    # cv.imshow("l" ,y )
    while True : 
        ret ,frame = webcam.read()
        
        if not ret : 
            print("Failed to capture")
            break
        
        frame = process_image(frame , face_detection)
        
        cv.imshow("Frame" , frame)
        
        if cv.waitKey(1) & 0xff ==ord('q') : 
            quit()
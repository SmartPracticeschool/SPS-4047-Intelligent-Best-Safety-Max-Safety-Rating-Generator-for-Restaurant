import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIAXOXZLFXGDMIJS5PX",
                        aws_secret_access_key="U1sRmQh2Y5MI+0lf5laHcCBoFPrLDIIUdFvHchRJ",
                        aws_session_token="FwoGZXIvYXdzEHkaDJbZKzISRCrrnZPyXCLHAUWa9EMIvCB618KcL+OkuH86w8X0NkivPehbzwsrgZHjpv8uL2RRAtlGP9VJ5461ziGakiIc097o5IutTNmYjG4e4N/cHx3sSE7AXXpk00IsntvMbZx3Luzqy5yaZbwi2rXtd1npW+clicC/7oKHm6MMKxJRkX2fbWdTMVmvH6COtWAbtkVOLUrEV53pQTWS5BX+2oXQ/BCe5CbvrylBbPQA4dWfFQW+LgEkASh66FT97Cug4WVXOpTnp9N3WqjKibIWCnr35VYoqN3s+gUyLaoUxmbNdKFEvdJiRDFr6ZoNtkm8ytBStjquswMcFCdwTofDWlzdZr/CKhVn+w==",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:512697773516:project/mask-detection/version/mask-detection.2020-09-10T18.47.28/1599743849377',Image={
            'Bytes':image1})

        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://9y0eeajx28.execute-api.us-east-1.amazonaws.com/count_mask?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()

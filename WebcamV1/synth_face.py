import numpy as np, cv2
def face():
    img = np.full((720,1280,3),210,np.uint8)
    cv2.ellipse(img,(640,380),(150,200),0,0,360,(190,170,160),-1)
    for cx in (585,695):
        cv2.ellipse(img,(cx,330),(34,18),0,0,360,(250,250,250),-1)
        cv2.circle(img,(cx,330),13,(70,55,45),-1); cv2.circle(img,(cx,330),5,(15,15,15),-1)
    cv2.ellipse(img,(640,380),(14,40),0,0,360,(170,150,140),-1)
    cv2.ellipse(img,(640,460),(48,16),0,0,360,(150,110,110),-1)
    return img

import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

if __name__ == "__main__":
    
    img_in = sys.argv[1]
    img_name = img_in[:-4]
    coords = sys.argv[2]
    img = cv2.imread(img_in)
    mask = np.zeros(img.shape[:2],np.uint8)
    
    pts = np.loadtxt(coords, dtype=np.uint64)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
        
    rect = (pts[0], pts[1], pts[2], pts[3])
    
    print(rect)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img1 = img*mask2[:,:,np.newaxis]
    
    background = img-img1
    
    background[np.where((background > [0,0,0]).all(axis =2))] = [255,255,255]
    
    final = background + img1
    
    plt.imshow(final),plt.colorbar(),plt.show()
    

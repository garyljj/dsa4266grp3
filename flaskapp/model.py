import cv2
import numpy as np
import skimage.measure

# Image stretching (constrast)
def clip_stretch(image, cmin, cmax):
    image = np.clip(image, cmin, cmax)
    image = (image - cmin) / (cmax - cmin) * 255.0
    return image.astype('uint8')

def mask_img(img):
    
    # Convert to grayscale
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Increase constrast + small blur to interpolate values
    mask = clip_stretch(mask, 175,220)
    mask = cv2.blur(mask,(3,3))

    # Blur/Clear middle portion
    h, w = mask.shape[:2]
    h, w= h//2, w//2
    midsize = 400
    mask[h-midsize:h+midsize,w-midsize:w+midsize] = cv2.blur(mask[h-midsize:h+midsize,w-midsize:w+midsize], (midsize, midsize))

    # Find edges
    mask = cv2.Canny(mask, 50, 50)

    # Reduce resolution by 6 times with max pixel value in attempts to join up boundaries
    origin_h, origin_w = mask.shape
    mask = skimage.measure.block_reduce(mask, (6,6), np.max)

    # Flood fill from center
    h, w = mask.shape
    mask_pad =  np.pad(mask, 1, 'minimum')
    cv2.floodFill(mask, mask_pad, seedPoint=(w//2,h//2-20), newVal=150)
    mask = np.where(mask == 150, 255, 0).astype('uint8')
    
    # Perform image closing to fill gaps
    mask = cv2.dilate(mask, np.ones((15,15)).astype('uint8'),iterations=1)
    mask = cv2.erode(mask, np.ones((10,10)).astype('uint8'),iterations=1)
   
    # Resize image back
    mask = cv2.resize(mask, (origin_w, origin_h))
    mask = np.where(mask==0,0,255).astype('uint8')
    
    # Perform image closing once more
    mask = cv2.dilate(mask, np.ones((40,40)).astype('uint8'),iterations=1)
    mask = cv2.erode(mask, np.ones((40,40)).astype('uint8'),iterations=1)
    
    # Covert to 0,1
    mask = np.where(mask==0,0,1).astype('uint8')
    
    for channel in range(3):
        img[:,:,channel] *= mask
    
    return img

import cv2 as cv
import numpy as np
import math as mt

# Methods to change the color model of images

def normalize_rgb(img):
  return cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float64)/255.0

def denormalize_rgb(img):
  n_img = (img*255).astype(int)
  return cv.cvtColor(n_img, cv.COLOR_RGB2BGR)

def rgb2hsl(img):

  height,width,depth = img.shape
  m = np.reshape(width*height,depth)
  size = width*height

  r = m[:,0]
  g = m[:,1]
  b = m[:,2]

  mx = np.max(m,axis=1)
  mn = np.min(m,axis=1)

  l = (mx+mn)/2
  s = np.zeros(l.shape)
  h = np.full(l.shape,mt.nan)

  for i in range(size):

    if mx[i] != mn[i]:
      continue

    max = mx[i]
    min = mn[i]
    d = max - min

    if l[i] <= 0.5:
      s[i] = d/(max+min)
    else:
      s[i] = d/(2-d)

    if r[i] == max:
      h[i] = (g[i]-b[i])/d
    elif g[i] == max:
      h[i] = (2+b[i]-r[i])/d
    else:
      h[i] = (4+r[i]-g[i])/d
    h[i] *= 60
    if h[i] < 0:
      h += 360

  HSL = np.array([h,s,l])
  return np.reshape(HSL.T,(height,width,depth))

def hsl2rgb(img):

  height,width,depth = img.shape
  m = np.reshape(width*height,depth)
  size = width*height

  h = m[:,0]
  s = m[:,1]
  l = m[:,2]

  r = np.zeros(size)
  g = np.zeros(size)
  b = np.zeros(size)

  for i in range(size):

    if s[i] == 0:
      if h[i] == mt.nan:
        r[i] = l[i]
        g[i] = l[i]
        b[i] = l[i]
      else:
        print('Error: chromatic case')
        raise ValueError

    if l[i] <= 0.5:
      v = l[i]*(1+s[i])
    else:
      v = l[i]+s[i]-(l[i]*s[i])
    
    if v != 0:

      min = 2*l[i]-v
      sv = (v-min)/v

      if h[i] == 360:
        hh = 0
      else:
        hh = h[i]/60
      sextant = mt.floor(hh)
      fract = hh - sextant
      vsf = v * sv * fract
      mid1 = min + vsf
      mid2 = v - vsf

      if sextant == 0:
        r[i] = v
        g[i] = mid1
        b[i] = min
      elif sextant == 1:
        r[i] = mid2
        g[i] = v
        b[i] = min
      elif sextant == 2:
        r[i] = min
        g[i] = v
        b[i] = mid1
      elif sextant == 3:
        r[i] = min
        g[i] = mid2
        b[i] = v
      elif sextant == 4:
        r[i] = mid1
        g[i] = min
        b[i] = v
      else:
        r[i] = v
        g[i] = min
        b[i] = mid2
  
  RGB = np.array([r,g,b])
  return np.reshape(RGB.T,(height,width,depth))
      
      



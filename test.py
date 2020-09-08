import cv2 as cv 
import utils
import result/mepso

images = ['flower.png','flower2.jpg','pepper.png','pepper2.png','peppers.JPG','simpleSBdiff.png','texture.png']

dir_in = 'data/'
dir_out = 'result/'
filein = dir_in + images[6]
img = cv.imread(filein)
h,w,d = img.shape

#cv.imshow('Image',img)

#cv.waitKey(0)

#Convert RGB to HSL

#rgb = utils.normalize_rgb(img)
#hsl = utils.rgb2hsl(rgb)

#Train MEPSO

model = mepso.MEPSO(img,w,h,10,3,10,'RGB',0.5,1,1,0.794,5,'rgb',20)
model.train()
result = model.get_image()

#Convert result hsl to rgb

#xrgb = utils.hsl2rgb(result)
#output = cv.cvtColor(result, cv.COLOR_RGB2BGR)

cv.imwrite(dir_out+images[6],result)

def train_rgb():

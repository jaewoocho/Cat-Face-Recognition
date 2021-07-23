import cv2
from math import atan2, degrees

# Reformat image size to 224
img_size = 224

# Creating margins for each image as a square
# Reformatting every image by the size of 224 by 224 square as every image is different in size

class resize:
def resize_img(im):
old_size = im.shape[:2] # old size in (height,width) format
ratio = float(img_size) / max(old_size)
new_size = tuple([int(x * ratio) for x in old_size])
# new_size should be in (width, height) format
im = cv2.resize(im, (new_size[1], new_size[0]))
delta_w = img_size - new_size[1]
delta_h = img_size - new_size[0]
top, bottom = delta_h // 2, delta_h - (delta_h // 2)
left, right = delta_w // 2, delta_w - (delta_w // 2)
new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
return new_im, ratio, top, left

# Function that helps on applying glasses on cat face
# Receives basic image, glasses, cat eye coordinates and adds glasses on cat face

class test:
# overlay function
# similar code as face recognition overlay code
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
bg_img = background_img.copy()
# convert 3 channels to 4 channels
if bg_img.shape[2] == 3:
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

if overlay_size is not None:
img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

b, g, r, a, = cv2.resize(img_to_overlay_t)

mask = cv2.medianBlur(a, 5)
h, w, _ = img_to_overlay_t.shape
roi = bg_img[int(y - h / 2):int(y + h / 2), int(x-w / 2):int(x + w / 2)]

img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

# convert 4 channels to 4 channels
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

return bg_img
# Calculates the different in coordinates between the cat eyes and glasses and rotates the glasses
def angle_between(p1, p2):
# Function that rotates the overlay image
xDiff = p2[0] - p1[0]
yDiff = p2[1] - p1[1]
return degrees(atan2(yDiff, xDiff))

#_____________________________________________________________________________________________________________________________________________
## import library
import random
import dlib, cv2, os
import pandas as pd
import numpy as np

##Loading the file path list
dirname = 'CAT_00'
base_path = '/Users/my/Downloads/cat/%s' % dirname
file_list = sorted(os.listdir(base_path))

for f in file_list:
if '.cat' not in f:
continue

#read landmarks
#as.matrix is now gone, to_numpy is the new one(Pandas package)
pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
landmarks = (pd_frame.to_numpy()[0][1:-1]).reshape((-1, 2)).astype(np.int)

#load image
img_filename, ext = os.path.splitext(f)

#Read image
img = cv2.imread(os.path.join(base_path, img_filename))

# visualize landmarks on cat features by drawing red circles on the image
for l in landmarks:
cv2.circle(img, center=tuple(l), radius=1, color=(0, 0, 255), thickness=2)
#Displaying the images
cv2.imshow('img', img)
if cv2.waitKey(0) == ord('q'):
break

#_____________________________________________________________________________________________________________________________________________

## os = system library
## pandas = Data manipulation / anaylze library
## helper = user library from helper.py

import random
import dlib, cv2, os
import pandas as pd
import numpy as np
from helper import resize


#As the cat dataset is divided by 7 folders the for function is applied
for i in range(7):
print(i, 'th preprocessing...')
dirname = 'CAT_0' + str(i)
# Receiving image from the path and the image is mixed it up
base_path = '/Users/my/Downloads/cat/%s' % dirname
file_list = sorted(os.listdir(base_path))
random.shuffle(file_list)

### Finally the img, lmks, bbs(bounding box / cat face area) is sent out as the output

dataset = {
'imgs': [],
'lmks': [],
'bbs': []
}

for f in file_list:
if '.cat' not in f:
continue

#read landmarks
#as.matrix is now gone, to_numpy is the new one
pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
landmarks = (pd_frame.to_numpy()[0][1:-1]).reshape((-1, 2))

#load image
img_filename, ext = os.path.splitext(f)
img = cv2.imread(os.path.join(base_path, img_filename))

#resize image and relocate landmarks
img, ratio, top, left = resize_img(img)
#recalculate edited landmarks
landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)
## face area allocate
bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

# dataset['imgs'].append(img)
# dataset['lmks'].append(landmarks.flatten())
#dataset['bbs'].append(bb.flatten())

## saving to file set
np.save('dataset/%s.npy' % dirname, np.array(dataset))

#_____________________________________________________________________________________________________________________________________________
### Training models takes a lot of time considering the Big O worst case
### If you don't have a GPU(ex) 1050ti ) pass this process and go straight to the test.py
import keras, datetime
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.applications import mobilenet_v2
import numpy as np
import tensorflow as tf

## Model that will find face features as an area from the image as mode is directed to 'bbs' (bounding box)
mode = 'bbs' # [bbs, lmks]
if mode is 'bbs':
##Sends the output image to x1, y 1 coordinates in the upper left and to x2, y2 coordinates in the lower right
output_size = 4
### If mdoe is 'lmks', finds landmarks from the face
elif mode is 'lmks':
### The output is 9 landmarks from the cat face(x,y is one pair which makes it 18)
output_size = 18


##If the mode is set at bbs, insert a normal image, if it is set at lmks, send in a landmark
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

## error occurred// Value Error: Object arrays cannot be loaded when allow_pickle=False
## First save the original np.load inside the np_load_old
np_load_old = np.load

# Change the original parameter
np.load = lambda * a, **k: np_load_old(*a, allow_pickle=True, **k)

print('dataloads start!')

data_00 = np.load('dataset/CAT_00.npy')
data_01 = np.load('dataset/CAT_01.npy')
data_02 = np.load('dataset/CAT_02.npy')
data_03 = np.load('dataset/CAT_03.npy')
data_04 = np.load('dataset/CAT_04.npy')
data_05 = np.load('dataset/CAT_05.npy')
data_06 = np.load('dataset/CAT_06.npy')

print('Finished Dataloads')
print('Initializing Data Preprocessing')


### The image is divided with 7 sets, from these 7 sets, 6 sets(00 ~ 06) are used to train and 1 set(07) is used as a test
### If Cross validation is required, add more sets or interchange other sets
x_train = np.concatenate((data_00.item().get('imgs'), data_01.item().get('imgs'), data_02.item().get('imgs'), data_03.item().get('imgs'), data_04.item().get('imgs'), data_05.item().get('imgs')), axis=0)
y_train = np.concatenate((data_00.item().get(mode), data_01.item().get(mode), data_02.item().get(mode), data_03.item().get(mode), data_04.item().get(mode), data_05.item().get(mode)), axis=0)


x_test = np.array(data_06.item().get('imgs'))
y_test = np.array(data_06.item().get(mode))

## Images have RBG values within 0 ~ 255, so they are divded by 255 to match 0 ~ 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

### For easier calculations, change the data set images into numpy arrays
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

# create input size
inputs = Input(shape=(img_size, img_size, 3))

print('Finished Data Preprocessing')
print('Initializing Model Building Process...')

# Error occurred/ Type Error: ('Invalid keyword argument: %s', 'depth_multiplier')
# Erase 'depth_multiplier'
# Place a dense layer behind mobilenet_v2 for transfer learning on the model

mobilenetv2_model = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()

print('Finished Data Preprocessing..')
print('Initializing Model Training Sequence')

# training
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=True,
validation_data=(x_test, y_test), verbose=1,
callbacks=[
TensorBoard(log_dir='logs/%s' % (start_time)),
ModelCheckpoint('./models/%s.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
]
)

print('Finished Model Training!')
#_____________________________________________________________________________________________________________________________________________
### Code that Displays final Results

import cv2, os
from keras.models import load_model
import numpy as np
from helper import resize, test

img_size = 224

# Assign glasses on the path from cats
base_path = 'sample'
file_list = sorted(os.listdir(base_path))

# This is the most important code; bringing the glasses that are used for the cat images
glasses = cv2.imread('images/glasses.png', cv2.IMREAD_UNCHANGED)

# Recall model and landmark trained models from path
bbs_model = load_model('models/bbs_1.h5')
lmks_model = load_model('models/lmks_1.h5')

print('Finished Loading Model')
print('Start Testing...')

# Applying glasses on every cat in the sample path; testing
for f in file_list:
if '.jpg' not in f:
continue

print('imread')

# Find the image
img = cv2.imread(os.path.join(base_path, f))
ori_img = img.copy()
result_img = img.copy()

print('Predict Bounding Box')

# predict bounding box
img, ratio, top, left = resize.resize_img(img)

inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

# Applies model that matches cat face
pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))


#Find the center of the face image
#Compute bounding box of original image
# 0.6 is multiplied to increase margins of the face as the initial face margins were a bit too tight
ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)

#Compute lazy bounding box for detecting landmarks
center = np.mean(ori_bb, axis=0)
face_size = max(np.abs(ori_bb[1] - ori_bb[0]))
new_bb = np.array([
center - face_size * 0.6,
center + face_size * 0.6
]).astype(np.int)
new_bb = np.clip(new_bb, 0, 99999)

print('Predict landmarks')

#predict landmarks
#Images are preprocessed to be applied on the model
face_img = ori_img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]
face_img, face_ratio, face_top, face_left = resize.resize_img(face_img)

face_inputs = (face_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

pred_lmks = lmks_model.predict(face_inputs)[0].reshape((-1, 2))

# compute landmark of original image
new_lmks = ((pred_lmks - np.array([face_left, face_top])) / face_ratio).astype(np.int)
ori_lmks = new_lmks + new_bb[0]

# Drawing the landmarks on the cat faces
# Visualize
cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=2)

for i, l in enumerate(ori_lmks):
cv2.putText(ori_img, str(i), tuple(l), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.circle(ori_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

print('Glasses Applied')

# Finding the coordinates of distance from the glasses and cat eyes to match the face
glasses_center = np.mean([ori_lmks[0], ori_lmks[1]], axis=0)
glasses_size = np.linalg.norm(ori_lmks[0] - ori_lmks[1]) * 2

angle = -test.angle_between(ori_lmks[0], ori_lmks[1])
M = cv2.getRotationMatrix2D((glasses.shape[1] /2, glasses.shape[0] / 2), angle, 1)
rotated_glasses = cv2.warpAffine(glasses, M, (glasses.shape[1], glasses.shape[0]))


## Rotate the glasses to match the cat face
try:
result_img = test.overlay_transparent(result_img, rotated_glasses, glasses_center[0], glasses_center[1],
overlay_size=(int(glasses_size), int(glasses.shape[0] * glasses_size / glasses.shape[1])))
except:
print('Failed to overlay Image')


# Visualize the results

cv2.imshow('img', ori_img)
cv2.imshow('result', result_img)
filename, ext = os.path.splitext(f)
cv2.imwrite('result/%s_lmks%s' % (filename, ext), ori_img)
cv2.imwrite('result/%s_result%s' % (filename, ext), result_img)

if cv2.waitKey(0) == ord('q'):
break

print('Finished Testing')

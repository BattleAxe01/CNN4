import tensorflow as tf

# load model
from tensorflow.keras.models import load_model

model = load_model("./model")

# inicialize variables
input_size = (128, 128)
path = "./dataset/predictions"

# image pre-processing
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# single prediction
# image pre-processing
from tensorflow.keras.preprocessing import image

img = image.load_img(path + "/cat.4005.jpg", target_size=input_size)
img = img_to_array(img)
img = np.expand_dims(img, axis=0)

res = model.predict_classes(img)
res = "cat" if res[0] == 0 else "dog"
print(res)

# # mass predictioin
# import cv2
# from imutils import paths

# img_path = paths.list_images("./dataset/predictions")
#
# orig = []
# pred = []
# for p in img_path:
#     orig.append(p)
#     img = cv2.imread(p)
#
#     img = cv2.resize(img, input_size)
#     img = img_to_array(img)
#
#     img = np.expand_dims(img, axis=0)
#
#     pred.append(img)
#
# # pred = np.array(pred)
#
# # predict model
# res = model.predict(pred)













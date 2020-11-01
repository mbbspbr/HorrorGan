import os
import numpy as np
from PIL import Image

#run training dataset through this to resize all images

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = 'dataset/'

images_path = IMAGE_DIR

training_data = []

print('resizing...')
for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    try:
        image = Image.open(path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        training_data.append(np.asarray(image))
    except Exception as e:
        print(e)
        print('Err on image: ', image)

training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

print('saving file...', training_data)
np.save('data.npy', training_data)

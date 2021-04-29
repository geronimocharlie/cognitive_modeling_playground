"""
Converts the training data into hdf5 file format, for efficiency during training.
"""
import h5py
import os
import numpy as np
import torchfile as tf
from PIL import Image

cub_dataset = "./CUB_200_2011/CUB_200_2011/"  # images
captions_dataset = "./cvpr2016_cub/word_c10/"  # captions
VocabSize = 5725 + 1  # Starts from 1 and ends at 5725, include start token also
start = 0  # 0 is start token
end = 1  # 1 is end token
maxcapslen = 30  # Max size of captions

# zip end compress
with h5py.File(os.path.join("dataset.hdf5"), "w") as h:
    h["capsperimage"] = 10
    images = h.create_dataset("images", (11788, 3, 256, 256), dtype="uint8") # size actually does not make too much sense as the resnet resizes it again to 224,224
    captions = h.create_dataset("captions", (11788, 10, maxcapslen + 2), dtype="int32")
    # saving classes in a separate numpy array bc of h5py compressing problems of ints (may be machine dependent)
    classes = np.zeros(shape=(11788,), dtype=np.int64)
    prev = ""
    last = 0
    captions[:] = 1

    with open(cub_dataset + "images.txt") as f:
        for i, line in enumerate(f):
            line = line.strip("\n").split(" ")[1]
            img = Image.open(cub_dataset + "images/" + line)
            img = img.resize((256, 256))
            img = np.asarray(img, dtype="int32")
            # The resnet accepts 3*w*w
            if img.shape != (256, 256, 3):  # A few of the images are black and white
                img = np.array([img, img, img])
                print(line)
            else:
                img = img.transpose(2, 0, 1)
            images[i] = img
            present = line.split("/")[0]

            if prev != present:

                prev = present
                temp = tf.load(
                    captions_dataset + line.split("/")[0] + ".t7",
                    force_8bytes_long=True,
                )
                temp = temp.transpose(0, 2, 1)

                captions[last : last + temp.shape[0], :, 0] = start
                captions[last : last + temp.shape[0], :, 1 : maxcapslen + 1] = temp
                last = last + temp.shape[0]

            classes[i]= int(line[:3])
        h["numcaptions"]=i*10
        print("num captions: ", i*10)
    print("done")

print("classes")
print(classes)
# sving classes seperateley
with open('classes.npy', 'wb') as f:
    np.save(f, classes)


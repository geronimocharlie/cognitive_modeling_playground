"""
cub:
Constructs the dataset from hdf5 file, which was created using datapreprocess script

coco:
Create the CoCoDataset and a DataLoader for it.
"""


from torch.utils.data import Dataset
import h5py
import nltk
nltk.download('punkt')
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import json
import torchvision.transforms as transforms


def get_coco_loader(transform,
               mode="train",
               batch_size=1,
               vocab_threshold=None,
               vocab_file="data_coco/vocab.pkl",
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc="."):

    """Return the data_cub loader.
    Parameters:
        transform: Image transform.
        mode: One of "train", "val" or "test".
        batch_size: Batch size (if in testing mode, must have batch_size=1).
        vocab_threshold: Minimum word count threshold.
        vocab_file: File containing the vocabulary.
        start_word: Special word denoting sentence start.
        end_word: Special word denoting sentence end.
        unk_word: Special word denoting unknown words.
        vocab_from_file: If False, create vocab from scratch & override any
                         existing vocab_file. If True, load vocab from from
                         existing vocab_file, if it exists.
        num_workers: Number of subprocesses to use for data_cub loading
        cocoapi_loc: The location of the folder containing the COCO API:
                     https://github.com/cocodataset/cocoapi
    """

    assert mode in ["train", "val", "test"], "mode must be one of 'train', 'val' or 'test'."
    if vocab_from_file == False:
        assert mode == "train", "To generate vocab from captions file, \
               must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file
    if mode == "train":
        if vocab_from_file == True:
            assert os.path.exists(vocab_file), "vocab_file does not exist.  \
                   Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, "cocoapi/images/train2014/")
        annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/captions_train2014.json")
        instances_file = os.path.join(cocoapi_loc, "cocoapi/annotations/instances_train2014.json")
    if mode == "val":
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data_cub."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, "cocoapi/images/val2014/")
        annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/captions_val2014.json")
        instances_file = os.path.join(cocoapi_loc, "cocoapi/annotations/instances_val2014.json")
    if mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data_cub."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, "cocoapi/images/test2014/")
        annotations_file = os.path.join(cocoapi_loc, "cocoapi/annotations/image_info_test2014.json")

    # COCO caption dataset
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder,
                          instances_file=instances_file)

    if mode == "train":
        # data_cub loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset,
                                      num_workers=num_workers,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      drop_last=False)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader


class CoCoDataset(data.Dataset):

    def __init__(self,
                 transform,
                 mode,
                 batch_size,
                 vocab_threshold,
                 vocab_file,
                 start_word,
                 end_word,
                 unk_word,
                 annotations_file,
                 vocab_from_file,
                 img_folder,
                 instances_file,
                ):

        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, annotations_file, vocab_from_file)
        self.vocab_size = len(self.vocab)
        self.img_folder = img_folder

        self.adaptive_pooling = torch.nn.AdaptiveAvgPool2d((224, 224))

        if self.mode == "train" or self.mode == "val":
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("number of images")
            print(len(self.ids))
           # print("now taking the first 1000") #<== comment out if you want to only train on 100
           # self.ids = self.ids[:1000]
           # print(len(self.ids))
            self.coco_instances = COCO(instances_file)
            self.ids_instances = list(self.coco_instances.anns.keys())


            self.categories_dict = self.coco_instances.cats
            with open('data_coco/coco_cats.npy', 'wb') as f:
                np.save(f, self.categories_dict)

            # coco_instances.anns is a dict with unique keys of dicts containing info on category
            self.classes = [self.coco_instances.anns[i]["category_id"] for i in self.ids_instances]
            self.num_classes = max(self.classes)
            print("number of unique classes")
            print(self.num_classes)

            print("Obtaining caption lengths...")
            all_tokens = [nltk.tokenize.word_tokenize(
                str(self.coco.anns[self.ids[index]]["caption"]).lower())
                for index in tqdm(np.arange(len(self.ids)))]

            self.caption_lengths = [len(token)+1 for token in all_tokens]
            self.max_caps_length = max(self.caption_lengths) + 1

        # If in test mode
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        # Obtain image and caption if in training or validation mode
        if self.mode == "train" or self.mode == "val":
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # getting instances
            class_k = torch.Tensor([self.classes[index]]).long()

            # if needed: name of supercategory
            #cat_name = self.categories_dict[class_k]["supercategory"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            # adaptive pooling used to bring all images to the same size
            image = self.adaptive_pooling(self.transform(image))

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])

            # filling up captions with end tokens to make them all of the same length
            #todo: please make padding more efficient
            while len(caption) < self.max_caps_length:
                caption.append(self.vocab(self.vocab.end_word))

            caption = torch.squeeze(torch.Tensor([caption]).long())
            capslen = torch.Tensor([self.caption_lengths[index]]).long() + 1 # + 1 to include end token

            # Return pre-processed image and caption tensors
            return image, caption, capslen, class_k

        # Obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # Return original image and pre-processed image tensor
            return orig_image, image

    def get_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == \
                                sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == "train" or self.mode == "val":
            return len(self.ids)
        else:
            return len(self.paths)

class CubDataset(Dataset):
    def __init__(self, transform):
        self.h = h5py.File("data_cub/dataset.hdf5", "r", driver="core")
        self.capsperimage = self.h["capsperimage"][()]
        self.images = np.array(self.h["images"])
        self.captions = np.array(self.h["captions"])
        self.maxcapslen = 30 + 3
        self.numcaptions = 117870 #self.h["numcaptions"][()]
        with open('data_cub/classes.npy', 'rb') as f:
            classes = np.load(f)
        print("classes", classes)
        print("captions",self.captions)
        self.class_k = classes

        self.transform = transform

    def __getitem__(self, i):
        img = torch.FloatTensor(self.images[i // self.capsperimage] / 255.0)
        img = self.transform(img)
        caption = torch.LongTensor(
            self.captions[i // self.capsperimage, i % self.capsperimage]
        )
        caplen = torch.LongTensor([self.maxcapslen - sum(caption == 1)])
        class_k = torch.LongTensor([self.class_k[i // self.capsperimage]])
        return img, caption, caplen, class_k

    def __len__(self):
        return self.numcaptions


if __name__ == "__main__":

    # testing cub
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataloader = CubDataset(transform=transforms.Compose([normalize]))
    for n, (i, c, l, ck) in enumerate(dataloader):
        print("image", i.size())
        print("ciption", c.size())
        print(c)
        if n==3:
            print("cub data succesful")
            break

    # testing coco
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataloader = get_coco_loader(transform=transforms.Compose([transforms.ToTensor(), normalize]), batch_size=32)
    start = 0
    end = 1
    print("max capslenght", max(dataloader.dataset.caption_lengths))
    print("num classes", dataloader.dataset.num_classes)

    for i, (img, c, class_k, capslen) in enumerate(dataloader):

        print("img", img.size())
        print("class", class_k)
        print("caption lenght", capslen)
        print("caption", c)

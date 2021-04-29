import numpy as np

if __name__ == "__main__":
    with open('coco_cats.npy', 'rb') as f:
        cats = np.load(f, allow_pickle=True)
    print(cats)
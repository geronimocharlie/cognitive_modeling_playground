# Context Aware Image Captioning
This repo is an attempt to build upon the work of Vedantam et al.  ([paper](https://arxiv.org/pdf/1701.02870.pdf) , [github](https://github.com/saiteja-talluri/Context-Aware-Image-Captioning)). The code is mostly copied, with a few alterations to work with the coco dataset and a modification of the beamsearch algorithm to incorporate multiple distractor classes. 

The Beamsearch can be executed with a few sample images, a pretrained model (refer to the 'Beamsearch' section) and the infos on the vocabulary. For the coco data this works out of the box, for the cub dataset at least the annotations must be downloaded. Please refere to CUB Dataset section for more infos. 
If you wish to run the training please refer to the section on 'Training'. 

## 1 Installation 
Create a conda environment using the provided 'environment.yml' file. If you wish to set everything up manually (or use another os than linux) refer to the file for the package specifications (found under 'pip'). 

```console
yourname@device:~/your_path_to…project$ conda env create -f environment.yml
yourname@device:~/your_path_to…project$ conda activate
```

## 2 Training (optional) 
### Preparing the data

#### COCO Dataset
To work with the Coco Dataset we use the [CocoAPI](https://github.com/cocodataset/cocoapi/tree/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9). (The repo is embedded in this repo, but if it is not cloned together with this repo please downolad manually and place it inside here). 
In addition to this API, download both the COCO images and annotations to run training. We use the Coco2014 data.
-Please download, unzip, and place the images ([train images](http://images.cocodataset.org/zips/train2014.zip) and [val images](http://images.cocodataset.org/zips/val2014.zip)) in: 'coco/images/'
-Please download and place the annotations ([train and val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip))  in 'coco/annotations'

For substantially more details on the API please see http://cocodataset.org/#download.

*Note*: I im unsure whether the pycocotools-fix pip packages makes everything work out of the box, in doupt the cocoAPI must be setup manually:
```console
(caic) yourname@device:~/your_path_to…project$ cd cocoapi/PythonAPI
(caic) yourname@device:~/your_path_to…project/cocoapi/PythonAPI$ make
```

#### CUB Datasest

##### Option A: preprocessed (recommended)
Please download the preprocessed dataset and place it under the 'data_cub' folder. You can download it from [here](https://drive.google.com/file/d/1WSlU_22In3sfHCGV6_KXlgfTDR6OR0-L/view?usp=sharing).
For the beamsearch to work, please also download the captions from [here](https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view) and place them under data_coco. (Needed for the vocabulary) 


##### Option B: preprocess yourself
In order to preprocess the cub dataset, you will need to download the images and annotations form [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)  and the captions from [here](https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view). Make sure the folders are named 'CUB_200_2011' and 'cvpr2016_cub' respectiveley. Place both folders under the 'data_cub' folder. Then excecute 'cub_preprocess.py'. 

*Note*: It might be the case that some of the bird names do not match. In the annotiations folder rename the following files in all subfolders: 
- cvpr2016_cub

      - Brewers_Blackbird into Brewer_Blackbird
      - Chuck_wills_Widow into Chuck_will_Widow
      - Brandts_Cormorant into Brandt_Cormorant
      - Heermanns_Gull into Heermann_Gull
      - Annas_Hummingbird into Anna_Hummingbird
      - Clarks_Nutcracker into Clark_Nutcracker
      - Scotts_Oriole into Scott_Oriole
      - Bairds_Sparrow inot Baird_Sparrow
      - Brewers_Sparrow into Brewer_Sparrow
      - Harriss_Sparrow into Harris_Sparrow
      - Henslows_Sparrow into Henslow_Sparrow
      - " Le_Conte_Sparrow
      - " Lincoln_Sparrow
      - Nelsons_Sparrow into Nelson_Sharp_tailed_Sparrow 
      - " Swainson_Warbler
      - " Wilson_Warbler
      - " Bewick_Wren

### Train
Once the data is prepared, please specify the hyperparameters (found in hyperparameters.py) to your liking. 

      - make sure to set the data_mode to 'cub' or 'coco' dependent on what data you have at hand
      - the model will be saved for later use in the models folder. Make sure it exists. 

To train the model to make it suitable for justification tasks (-> dependent on class) run:
```console
(caic) yourname@device:~/your_path_to…project$ python train_justify.py
```

## Beamsearch

To execute the beamsearch you will need to download or create a pretrained model. A pretrained model on a subset of coco (1000 images of the validation set of coco) can be found [here](https://drive.google.com/file/d/1FOQF8XwDT1DpxaBaWi6vEnm-cTP5hVLR/view?usp=sharing)
Please make sure to adjust the path to your pretrained model (checkpoint_j for justifications) in the hyperparameters.py script. You can also alter the lambda parameter and beam size there. 

Also testing images will be needed. You can use the ones provided in the test_images folder or place your own there. 

If running on the coco dataset, you can get an overview of the existing classes by running: 
```console
(caic) yourname@device:~/your_path_to…project$ python data_coco/ontology_feeder.py
```

To excecute the original beamsearch algorithm:

    Justification: python beamsearch.py cj target_image_path target_class distractor_class
    e.g. python beamsearch.py cj 'test_images/moto.jpg' 4 3

    Context agnostic captioning: python beamsearch.py c image_path
   
    Discrimination: python beamsearch.py cd target_image_path distractor_image_path


To perform beamsearch on the justification task for **multiple** distractor classes:

```console
(caic) yourname@device:~/your_path_to…project$ python beamsearch_j_multi_d.py  target_image_path target_class distractor_class_1 distractor_class_2 ...
```
example: 

```console
(caic) yourname@device:~/your_path_to…project$ python beamsearch_j_multi_d.py 'test_images/moto.jpg' 4 2 3 5
```

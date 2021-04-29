# Context Aware Image Captioning
This repo is an attempt to build upon the work of  (paper: , github:). The code is mostly copied, with a few alterations to incorporate the coco dataset and a modification of the beamsearch algorithm to incorporate multiple distractor classes. 

The Beamsearch can be executed with a few sample images and a pretrained model, thus downloading and preparing the dataset is not possible. For more information see the section on 'Beamsearch'. 
If you wish to run the training please refer to the section on 'Training'. 


## Training
### Preparing the data

#### COCO Dataset
To work with the Coco Dataset we use the [CocoAPI](https://github.com/cocodataset/cocoapi/tree/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9). (The repo is embedded in this repo, but if it is not cloned together with this repo please downolad manually and place it inside here). 
In addition to this API, please download both the COCO images and annotations to run training. We use the Coco2014 data.
-Please download, unzip, and place the images ([train images](http://images.cocodataset.org/zips/train2014.zip) and [val images](http://images.cocodataset.org/zips/val2014.zip)) in: 'coco/images/'
-Please download and place the annotations ([train and val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip))  in 'coco/annotations'

For substantially more details on the API please see http://cocodataset.org/#download.

#### CUB Datasest

##### Option A: preprocessed (recommended)
Please download the preprocessed dataset and place it under the 'data_cub' folder. You can download it from [here](https://drive.google.com/file/d/1WSlU_22In3sfHCGV6_KXlgfTDR6OR0-L/view?usp=sharing)


##### Option B: preprocess yourself
In order to preprocess the cub dataset, you will need to download the images and annotations form [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)  and the captions from [here](https://drive.google.com/file/d/0B0ywwgffWnLLZW9uVHNjb2JmNlE/view). Mawe sure the foleders are named 'CUB_200_2011' and 'cvpr2016_cub' respectiveley. Place both folders under the 'data_cub' folder. Then excecute 'cub_preprocess.py'. 

*Note*: It might be the case that some of the bird names do not mathc. In the annotiations folder rename the following files in all subfolders: 
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

### Train. 
Once the data is prepared, please specify the hyperparameters (found in hyperparameters.py) to your liking. 
- make sure to set the data_mode to 'cub' or 'coco' dependent on what data you have at hand
- the model will be saved for later use in the models folder. Make sure it exists. 

To train the model to make it suitable for justification tasks (-> dependent on class) rund:
$ python train_justify.py


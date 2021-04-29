## Preparing the data

### CUB Datasest

#### Option A: preprocessed 
Please download the preprocced dataset and place it under the 'data_cub' folder. You can download it from here: 
https://drive.google.com/file/d/1WSlU_22In3sfHCGV6_KXlgfTDR6OR0-L/view?usp=sharing


#### Option B: preprocces yourself
In order do preprocess the cub dataset, you will need to download the images form here and the annotiations from here. Mawe sure the foleders are named 'CUB_200_2011' and 'cvpr2016_cub' respectiveley. Place both folders under the 'data_cub' folder. Then excecute 'cub_preprocess.py'. 

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
      - Nelsons_Sparrow into Nelson_Sharp_tailed_Sparrow (now you guys are just mocking me right?)
      - " Swainson_Warbler
      - " Wilson_Warbler
      - " Bewick_Wren

# BTG-Net++: Enhanced Bidirectional Task-Guided Network for Few-Shot Fine-Grained Image Classification

## This work is an extension of an ACM MM 2024 paper: Bidirectional Task-Guided Network for Few-Shot Fine-Grained Image Classification

## Data Preparation

The following datasets are used in our paper:

Stanford Dogs: [Dataset Page](http://vision.stanford.edu/aditya86/ImageNetDogs/)

Stanford Cars: [Dataset Page](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view)

CUB_200_2011: [Dataset Page](https://www.vision.caltech.edu/datasets/cub_200_2011/)

iNaturalist2017 : [Dataset Page](https://github.com/visipedia/inat_comp/tree/master/2017)

Please proceed with the setting up data by referring to [FRN Github](http://github.com/Tsingularity/FRN#setting-up-data).



## Usage

### Requirement
All the requirements to run the code are in requirements.txt.

You can download requirements by running below script.
```
pip install -r requirements.txt
```

<!-- ### Dataset directory
Change the data_path in config.yml.
```
dataset_path: #your_dataset_directory
```
 -->

### Train and test
Running the shell script ```run.sh``` will train and evaluate the model with hyperparameters matching our paper.



# Equitable Skin Disease Prediction Using Transfer Learning and Domain Adaptation

### Data Download
- Download DDI data from [here](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965) 
- Follow our train-val-test split as described in the paper and create thress sub-directories train, val, test and put images in each label directory inside those sub-directories.
- The file hiererchy should look like this:
  - train
    - benign
    - malignant
  - val
    - benign
    - malignant
  - test
    - benign
    - malignant

### Download Pre-trained weights
- Download corresponding model weights from the method papers before running the finetuning scripts

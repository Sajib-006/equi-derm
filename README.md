# Equi-Derm
## Equitable Skin Disease Prediction Using Transfer Learning and Domain Adaptation [[arxiv](https://arxiv.org/abs/2409.00873)]
**Abstract:** 
In the realm of dermatology, the complexity of diagnosing
skin conditions manually requires the expertise of dermatologists. Accurate identification of various skin ailments,
ranging from cancer to inflammatory diseases, is paramount.
However, existing artificial intelligence (AI) models in dermatology face challenges, particularly in accurately diagnosing diseases across diverse skin tones, with a notable performance gap in darker skin. Furthermore, the scarcity of publicly available and unbiased data sets hampers the development of inclusive AI diagnostic tools. To address the challenges in accurately predicting skin conditions across diverse
skin tones, we employ a transfer-learning approach that capitalizes on the rich and transferable knowledge from various
image domains. Our method integrates multiple pre-trained
models from a wide range of sources, including general and
specific medical images, to improve the robustness and inclusiveness of the skin condition predictions. We rigorously
evaluated the effectiveness of these models using the Diverse Dermatology Images (DDI) dataset, which uniquely encompasses both underrepresented and common skin tones,
making it an ideal benchmark for assessing our approach.
Among all methods, Med-ViT emerged as the top performer
due to its comprehensive feature representation learned from
diverse image sources. To further enhance performance, we
performed domain adaptation using additional skin image
datasets such as HAM10000. This adaptation significantly
improved model performance across all models.

![Model Architecture](figure/model_archi.pdf)


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

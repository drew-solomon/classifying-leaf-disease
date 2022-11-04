# Classifying Cassava Leaf Disease
 This project classifies cassava leaf diseases using deep learning, leveraging image augmentation and transfer learning with CNN models.\
_(Midterm Project for DATA 2040 - Deep Learning @ Brown University Spring 2021)_

## Table of Contents
* [Project Description](#project-description)
* [Methods](#methods-used)
* [Technologies Used](#technologies-used)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Deliverables](#deliverables)
* [Contributing Members](#contributing-members)
* [Acknowledgements](#acknowledgements)

## Project Description
- **Background**: Cassava is a key crop for food security across Sub-Saharan Africa. Yet, viral diseases threaten cassava yields, and are costly to detect manually.
- As a midterm group project for DATA2040, this project fine-tunes a series of CNN models to accurately detect cassava leaf diseases, using image augmentation and transfer learning to increase classification accuracy.
- **Why?**: Fine-tuned deep learning models may help identify diseased cassava plants more efficiently and ultimately prevent crop loss.
- The data consists of 21,367 labeled images of cassava leaves belonging to five different categories - four different disease categories and one category for healthy plants. The images were crowdsourced from farmers in Uganda and labeled by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. The data and task was made available as a Kaggle [competition](https://www.kaggle.com/c/cassava-leaf-disease-classification). 
- **Result**: Using image augmentation and transfer learning we increased our accuracy from a baseline 61.5% (majority classifier) to 80% (DenseNet201 model).

## Methods Used
- EDA
- Image data augmentation
- Deep learning (CNN)
- Transfer learning
- Fine-tuning
- Hyperparameter tuning

## Technologies Used
- Python (3.8)
- TensorFlow (2.4.1)
- Keras (2.4.0)
- Scikit-learn (0.24.0)
- Pandas (1.2.1)
- Numpy (1.19.2)

## Screenshots


#### EDA - class balance 
| ![Class balance ](https://i.postimg.cc/Qxs7F7vQ/Screenshot-2022-11-02-at-17-42-23.png)
|:--:|
|*Cassava disease class balance (normalized). Due to the data imbalance, we used a StratifiedKFold split to preserve class ratios in each fold.*|

#### EDA - image data by category
|![Image data by category](https://i.postimg.cc/Gtp1fqHP/Screenshot-2022-11-02-at-17-44-09.png)
|:--:|
|*Cassava leaf images by disease category. The images vary by angle, lighting, and position.*|

#### Model training - baseline VGG16 model
|![Baseline model](https://i.postimg.cc/R09CR1Qd/Baseline-model.png)
|:--:|
|*Validation and training accuracy for our baseline VGG16 model over 16 epochs.*|

#### Model training - DenseNet model
|![DenseNet model](https://i.postimg.cc/QdqCvnMw/densenet.png)
|:--:|
|*Validation and training accuracy for fine-tuned DenseNet model with a ReduceLROnPlateau schedule over 31 epochs.*|



## Setup

To read the project code as a Jupyter/IPython notebook, click on the [RAD_Final_Blog_Post_3.ipynb](https://github.com/drew-solomon/classifying-leaf-disease/blob/main/RAD_Final_Blog_Post_3.ipynb) file here to open it in your browser. 

The project notebook can also be opened and run in Google Colab (with GPU). To download the Kaggle Cassava Leaf Disease Classification [dataset](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data), create a Kaggle account and create an API token. Then, replace the Kaggle username and key in the following code cell:
[![Kaggle username and key cell](https://i.postimg.cc/J43PwWBR/Screenshot-2022-11-02-at-19-26-40.png)](https://postimg.cc/hfj9x6rY)



## Deliverables

We described our project process - from data exploration and augmentation to transfer learning and model fine-tuning - in the following blog posts:

- [Blog - Part 1](https://roma-coffin.medium.com/categorizing-cassava-leaf-diseases-dbd08fcc671)
- [Blog - Part 2](https://roma-coffin.medium.com/using-deep-learning-to-classify-cassava-leaf-diseases-part-2-1321cd61d46)
- [Blog - Part 3](https://roma-coffin.medium.com/categorizing-cassava-leaf-diseases-part-3-bbf6d002d3d8)

## Contributing Members
- [Annie Phan](https://github.com/annieptba)
- [Roma Coffin](https://github.com/romacoffin)
- [Drew Solomon](https://github.com/drew-solomon) (me)


## Acknowledgements

- This project task and data was sourced from Kaggle’s Cassava Leaf Disease Classification [competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview).
- As such, many thanks to the Makerere Artificial Intelligence (AI) Lab at Makerere University in Uganda - who apply AI and data science to real-word challenges, as well as to the experts and collaborators from National Crops Resources Research Institute (NaCRRI) for assisting in preparing this dataset.
- All of our sources for our EDA and model development are listed in our blog posts ([1](https://roma-coffin.medium.com/categorizing-cassava-leaf-diseases-dbd08fcc671), [2](https://roma-coffin.medium.com/using-deep-learning-to-classify-cassava-leaf-diseases-part-2-1321cd61d46), and [3](https://roma-coffin.medium.com/categorizing-cassava-leaf-diseases-part-3-bbf6d002d3d8)) in the *Sources Used* section.
- Many thanks to Annie and Roma for your collaboration!

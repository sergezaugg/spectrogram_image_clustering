# CLUSTER IMAGES WITH DNN FEATURES AND DIM REDUCTION

### Overview
* This is a Streamlit dashboard to cluster-analyse data from images
* Features were pre-extracted from images offline with a script ```extract_features.py```
* The resulting npz file(a) must be loaded to a Kaggle dataset [Examlpe Kaggle Dataset](https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
* Third, the Streamlit process in started ```streamlit run stmain.py``` (e.g. locally of on https://share.streamlit.io)
* The path to Kaggle dataset must be adjusted in the Streamlit code.
* Thats all, now the dashboard is active.
* See the deployed version [here](https://food-image-clustering.streamlit.app)

### Data
* This project is based on images from **Food Classification Dataset** shared on Kaggle by Bjorn.
* https://www.kaggle.com/datasets/bjoernjostein/food-classification
* Over 9300 hand-annotated images with 61 classes

### Feature extraction (image to vector)
* Features extracted with image classification models pre-trained with the Imagenet datataset.
* Details see on [PyTorch documentation](https://docs.pytorch.org/vision/main/models.html)
* As features we used output from last linear layer of image CNNs: 1000 continuous values. 
* These CNNs were trained specifically for the Imagenet classes, let's hope the feature are informative for our task.
* Pre-extracted features available [here](https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)

### Clustering, dim-reduction, and visualization
* First features are dim-reduced with UMAP
* Second cluster-IDs are obtained with DBSCAN (unsupervised -> without using the ground truth)
* Third, cluster-IDs and ground truth are compared visually and with metrics.

### Dependencies / Intallation
* Developed under Python 3.12.8
* Make a fresh venv!
* For Streamlit deployment only
```bash 
pip install -r requirements.txt
```
* For feature extraction (PyTorch / GPU) and Streamlit deployment 
```bash 
pip install -r req_torchcuda.txt
```

### Usage 
*  To extract features, see **extract_features.py**
*  Start dashboard
```bash 
streamlit run stmain.py
```



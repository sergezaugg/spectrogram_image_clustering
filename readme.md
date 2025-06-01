# CLUSTER IMAGES WITH DNN FEATURES AND DIM REDUCTION


### Overview - backend code
* Backend code for pre-computing feature extraction and dim-reduction 
* ideally with a GPU manchine
* Code is in subdir ```./pt_extract_features```
* Short example in ```./pt_extract_features/main.py```

### Overview - dashboard
* This is a Streamlit dashboard to cluster-analyse images of spectrograms
* Features were pre-extracted from images offline with a script ```01_extract.py```
* The resulting npz file(a) must be loaded to a Kaggle dataset [Examlpe Kaggle Dataset](https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
* Third, the Streamlit process in started ```streamlit run st_dashboard/stmain.py``` (e.g. locally of on https://share.streamlit.io)
* The path to Kaggle dataset must be adjusted in the Streamlit code.
* Thats all, now the dashboard is active.
* See the deployed version [here](https://food-image-clustering.streamlit.app)

### Data
* tbd


### Feature extraction (image to vector)
* Features extracted with image classification models pre-trained with the Imagenet datataset.
* Details see on [PyTorch documentation](https://docs.pytorch.org/vision/main/models.html)
* As features we used output edit!


### Clustering, dim-reduction, and visualization
* First features are dim-reduced with UMAP
* Second cluster-IDs are obtained with DBSCAN (unsupervised -> without using the ground truth)

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
streamlit run st_dashboard/stmain.py
```



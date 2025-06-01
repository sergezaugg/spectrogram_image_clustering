
# CLUSTER SPECTROGRAMS WITH DNN FEATURES AND DIM REDUCTION

### Overview 
* This codebase has two parts: 
* The **backend code** to pre-compute features from spectrogram images (to be stored in a public Kaggle dataset)
* The **frontend code** that uses the pre-computed features to feed a Streamlit dashboard 

### Source data
* Acoustic recordings are from [xeno-canto](https://xeno-canto.org/)
* Standardized acoustic data preparation was performed with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)  
* In a nutshell: MP3 converted to WAVE, resampled, transformed to spectrograms, and stored as RGB images.
* Images can then be fed into the backend code for feature extractions

### Backend code
* Backend code is for feature extraction and dim-reduction 
* Code is in this subdir ```./pt_extract_features```
* Main functionality is in this module: ```utils_ml```
* Short example in ```main.py```
* There are 3 flat scripts used to perform the extraction: ```01_ 02_ 03_```
* Full and reduced-dim features as stored as NPZ files
* NPZ file are then stored on a Kaggle dataset [example](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01) where the frontend will fetch them.

### Frontend code
* This is a Streamlit dashboard to cluster-analyse images of spectrograms
* Features were pre-extracted from images offline (see above)
* Third, the Streamlit process in started ```streamlit run st_dashboard/stmain.py``` (e.g. locally of on https://share.streamlit.io)
* The path to Kaggle dataset must be adjusted in the Streamlit code.
* Thats all, now the dashboard is active.
* See the deployed version [here](https://spectrogram-image-clustering.streamlit.app)

### Dependencies / Intallation
* Developed under Python 3.12.8
* For feature extraction with PyTorch (and GPU) ```pip install -r req_torchcuda.txt```
* For Streamlit deployment ```pip install -r requirements.txt```

### Machine Learning
* Please find detes [here](https://spectrogram-image-clustering.streamlit.app/page01)




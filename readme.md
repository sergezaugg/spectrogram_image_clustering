
# Interactive app for pre-annotation of spectrogram images aided by deep-leaning based clustering

### Overview 
* This is a Streamlit dashboard to cluster-analyse images of spectrograms
* Large collection of short spectrogram images are processed with unsupervised clustering
* The user can tune a few parameters to find consistent clusters
* Consistent clusters can be pooled and then downloaded as a dummy-coded data frame

### Source data preparation
* Acoustic recordings are from [xeno-canto](https://xeno-canto.org/)
* Standardized acoustic data preparation was performed with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)  
* Feature extraction was performed with [this tool](https://github.com/sergezaugg/spectro_aec_clust) and [this tool](https://github.com/sergezaugg/spectro_image_feature_extract) 
* In a nutshell: MP3 converted to WAVE, resampled, transformed to spectrograms, stored as RGB images, features extracted.
* Images and extracted features are stored on a kaggle datasets used by this dashboard.

### Usage
* The Streamlit process in started ```streamlit run st_dashboard/stmain.py``` (e.g. locally of on https://share.streamlit.io)
* The path to Kaggle dataset must be adjusted in the Streamlit code.
* See the deployed version [here](https://spectrogram-image-clustering.streamlit.app)

### Dependencies / Intallation
* Developed under Python 3.12.8
* Make a fresh venv!
```bash 
pip install -r requirements.txt
```

### Machine Learning
* Please find detes [here](https://spectrogram-image-clustering.streamlit.app/page01)




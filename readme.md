# Interactive app for pre-annotation of spectrogram images aided by deep-leaning based clustering

## Overview

- An interactive [Streamlit](https://streamlit.io/) dashboard for pre-annotation of spectrogram images using deep learning features.
- Built for exploration and semi-automated labeling of acoustic datasets.
- Explore large collections of unselected spectrogram images.
- Tune clustering parameters to find consistent groups.
- Pool and download these groups as dummy-coded data frames for downstream annotation or machine learning.
- See the deployed version [here](https://spectrogram-image-clustering.streamlit.app)

## Features

- **Data Source Selection:** Choose from pre-extracted datasets with spectrograms features.
- **Feature Extraction:** Uses features from Image DNNs (IDNN) and Spectrogram AutoEnCoders (SAEC).
- **Interactive Clustering:** Intuitive DBSCAN-based clustering with adjustable parameters in 2 to 16 dims.
- **Pre-partition:** Data pre-partitionning with K-means to reduce memory consumption by DBSCAN  
- **Visualization:** UMAP-based 2D scatterplots for cluster previews.
- **Image Pooling:** Assign several consistent clusters to a class and export as CSV for annotation.

## Data Preparation to feed this app

- Acoustic recordings sourced from [xeno-canto](https://xeno-canto.org/).
- Preprocessing of acoustic data:
  - [xeno_canto_organizer](https://github.com/sergezaugg/xeno_canto_organizer)
  - In a nutshell: MP3 converted to WAVE, resampled, transformed to spectrograms, stored as RGB images.
- Feature extraction performed in two modalities:
  - [Features from Image DNNs (IDNN)](https://github.com/sergezaugg/feature_extraction_saec)
  - [Features from Spectrogram AutoEnCoders (SAEC)](https://github.com/sergezaugg/feature_extraction_idnn)
  - Please find the ML detes [here](https://spectrogram-image-clustering.streamlit.app/page01)
- Spectrograms and features are stored on Kaggle datasets, accessed directly by the app.

## Usage

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2. **Run the app:**
    ```bash
    streamlit run st_dashboard/stmain.py
    ```
3. **Follow the sidebar instructions:**
    - Select a data source and feature sets.
    - Adjust clustering parameters.
    - Explore, pool, and export clusters.

## File Structure

- `st_dashboard/` — Main Streamlit app and utility modules.
- `pics/` — Images for the UI.
- `.streamlit/` — Streamlit configuration.
- `requirements.txt` — Python dependencies.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- Recordings: [xeno-canto](https://xeno-canto.org/)
- Feature extraction and clustering: See linked GitHub projects above.
- Created by [Serge Zaugg](https://www.linkedin.com/in/dkifh34rtn345eb5fhrthdbgf45/).

---

For more details and a live demo, see the [deployed app](https://spectrogram-image-clustering.streamlit.app)
#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run st_dashboard/stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss
import numpy as np
import gc
from utils import data_source_format
gc.collect()

st.set_page_config(layout="wide")

# data objects
if 'dapar' not in ss:
    ss['dapar'] = {
        'feat_path' : 'empty',
        'imgs_path' : 'empty',
        'dataset_name' :  'empty',
        'clusters_pred_str' : np.array([]),
        'im_filenames' : np.array([]),
        'li_npz' : 'empty', # list of available files
        'X2D' : np.array([]),
        'X_dimred' : np.array([]),
        'df_meta' : np.array([]),
        'kgl_datasource' : 'empty',
        'image_pool' : list(),
        'df_prelim_labels' : np.array([]),
        }

# user provided parameters
if 'upar' not in ss:
    ss['upar'] = {
        'umap_n_neighbors' : 10,
        'umap_n_dims_red' : 8,
        'skip_umap' : False,
        'dbscan_eps' : 0.46,
        'dbscan_min_samples' : 20,
        'exclude_non_assigned' : False,
        'datsou' : "spectrogram-clustering-01",
        }

with st.sidebar:  
    st.header(''':primary[**Unsupervised clustering for pre-annotation of spectrogram datasets (beta)**]''')
    # st.header('''(beta)''')
    st.info('Data source: ' + data_source_format(ss['upar']['datsou']))
    st.info('Features from: ' + ("_".join(ss['dapar']['dataset_name'].split("_")[4:])[0:22] )) # yes, APITA
    st.text("")
    st.markdown('''QUICK GUIDE  
                (1) Select a data source    
                (2) Activate features     
                (3) Tune DBSCAN params    
                (4) Get adequate Nb clusters*  
                (4) Search consistent clusters   
                (5) Assign clusters to pool     
                (7) Save pool as a class variable    
                (8) Repeat 2-7 to make more variables      
                ''')
   
    # logos an links
    st.text("")
    c1,c2=st.columns([80,200])
    c1.image(image='pics/z_logo_orange.png', width=65)
    c2.markdown(''':primary[v0.9.13 (beta)]  
    :primary[Created by]
    :primary[[Serge Zaugg](https://www.linkedin.com/in/dkifh34rtn345eb5fhrthdbgf45/)]    
    :primary[[Pollito-ML](https://github.com/sergezaugg)]
    ''')
    st.logo(image='pics/z_logo_orange.png', size="large", link="https://github.com/sergezaugg")

p01 = st.Page("page01.py", title="ML-summary")
p02 = st.Page("page02.py", title="Cluster spectrograms")
p03 = st.Page("page03.py", title="Select data")
p04 = st.Page("page04.py", title="Data info and credits")
p05 = st.Page("page05.py", title="Image pool")
pss = st.Page("page_ss.py", title="(Dev diagnostics)")
pg = st.navigation([p03, p02, p05, p01, p04,   
                    # pss
                    ])
pg.run()


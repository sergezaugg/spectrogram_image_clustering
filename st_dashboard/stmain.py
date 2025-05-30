#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run st_dashboard/stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss
import numpy as np
import gc
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
        }

# user provided parameters
if 'upar' not in ss:
    ss['upar'] = {
        'umap_n_neighbors' : 10,
        'umap_n_dims_red' : 8,
        'skip_umap' : False,
        'dbscan_eps' : 0.46,
        'dbscan_min_samples' : 20,
        'exclude_non_assigned' : False
        }

with st.sidebar:
    
    st.info('Selected: ' + ss['dapar']['dataset_name'])
    st.header(''':primary[**Unsupervised clustering of spectrograms with features from pre-trained image models**]''')
    st.header("")
    st.markdown('''QUICK GUIDE''')
    st.text("(1) Select a dataset")
    st.text("(2) Tune DBSCAN params")
    st.text("(3) Explore scatterplot")
    st.text("(4) Check cluster contents")

    # logos an links
    st.header(""); st.header(""); st.header("")
    c1,c2=st.columns([80,200])
    c1.image(image='pics/z_logo_orange.png', width=65)
    c2.markdown(''':primary[v0.9.8]  
    :primary[Created by]
    :primary[[Serge Zaugg](https://github.com/sergezaugg)]''')
    st.logo(image='pics/z_logo_orange.png', size="large", link="https://github.com/sergezaugg")



p01 = st.Page("page01.py", title="Summary")
p02 = st.Page("page02.py", title="Analyse")
p03 = st.Page("page03.py", title="Select dataset")
p04 = st.Page("page04.py", title="Data info/credits")
pss = st.Page("page_ss.py", title="(Dev diagnostics)")
pg = st.navigation([p03, p02, p01, p04, 
                    # pss
                    ])
pg.run()


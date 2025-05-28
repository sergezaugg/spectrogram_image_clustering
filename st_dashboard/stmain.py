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
        }

# user provided parameters
if 'upar' not in ss:
    ss['upar'] = {
        'umap_n_neighbors' : 10,
        'umap_n_dims_red' : 8,
        'skip_umap' : False,
        'dbscan_eps' : 0.501,
        'dbscan_min_samples' : 10,
        'exclude_non_assigned' : False
        }

with st.sidebar:
    st.info('Selected: ' + ss['dapar']['dataset_name'])
    st.header(''':primary[**CLUSTER SPECTROGRAMMS**]''')
    st.text("v0.2.0")
    st.markdown(''':primary[QUICK GUIDE]''')
    st.text("(1) Select a dataset")
    st.text("(2) Tune DBSCAN params")
    st.text("(3) Explore plots and figures")

p01 = st.Page("page01.py", title="Summary")
p02 = st.Page("page02.py", title="Analyse")
p03 = st.Page("page03.py", title="Select dataset")
pss = st.Page("page_ss.py", title="(Debug diagnostics)")
pg = st.navigation([p03, p02, p01, pss])
pg.run()


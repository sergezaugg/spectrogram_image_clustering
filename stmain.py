#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss
import numpy as np

st.set_page_config(layout="wide")

if 'dapar' not in ss:
    ss['dapar'] = {
        'feat_path' : 'empty',
        'imgs_path' : 'empty',
        'npdata'    : 'empty',
        'dataset_name' :  'empty',
        'X' : np.array([]),
        'clusters_pred_str' : np.array([]),
        'im_filenames' : np.array([]),
        }

if 'upar' not in ss:
    ss['upar'] = {
        'umap_n_neighbors' : 10,
        'umap_n_dims_red' : 16,
        'skip_umap' : False,
        'dbscan_eps' : 0.501,
        'dbscan_min_samples' : 10,
        }

with st.sidebar:
    st.info(ss['dapar']['dataset_name'])
    st.header(''':blue[**CLUSTER SPECTROGRAMMS**]''')
    st.text("v0.0.0")
    st.markdown(''':blue[QUICK GUIDE]''')
    st.text("(0) Select a dataset")
    st.text("(1) Choose UMAP params")
    st.text("(2) Tune DBSCAN params")
    st.text("(3) Explore metrics & plots")
    st.markdown(''':blue[COMPUTATION SPEED]''')
    st.text("UMAP can take a few minutes while DBSCAN takes a few seconds; if values cached it is faster.")

p01 = st.Page("page01.py", title="Summary")
p02 = st.Page("page02.py", title="Analyse")
p03 = st.Page("page03.py", title="Select dataset")
pss = st.Page("page_ss.py", title="(Debug diagnostics)")
pg = st.navigation([p03, p02, p01, pss])
pg.run()


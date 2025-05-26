#--------------------             
# Author : Serge Zaugg
# Description : Main interactive streamlit page
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import gc
from utils import dim_reduction_for_2D_plot, make_sorted_df, make_scatter_3d_plot
gc.collect()
 
X2D_scaled_3d = dim_reduction_for_2D_plot(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'], n_components = 3)
df_true_3d = make_sorted_df(cat = ss['dapar']['clusters_true'], cat_name = 'True class', X = X2D_scaled_3d)
df_pred_3d = make_sorted_df(cat = ss['dapar']['clusters_pred_str'], cat_name = 'Predicted cluster', X = X2D_scaled_3d)
fig01_3d = make_scatter_3d_plot(df = df_true_3d, cat_name = 'True class', title = "Ground truth")
fig02_3d = make_scatter_3d_plot(df = df_pred_3d, cat_name = 'Predicted cluster', title = "Predicted clusters")
   
# show plots 
c01, c02,  = st.columns([0.5, 0.5])
with c01:
    with st.container(border=True):  
        st.plotly_chart(fig01_3d, use_container_width=False, theme=None)
with c02:
    with st.container(border=True):  
        st.plotly_chart(fig02_3d, use_container_width=False, theme=None)

gc.collect()



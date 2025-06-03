
#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import streamlit as st
from streamlit import session_state as ss
import gc

gc.collect()

st.text("Data frame with clustering-based labels can be downloaded here")
st.write(ss['dapar']['df_prelim_labels'].shape)
st.dataframe(ss['dapar']['df_prelim_labels'], width=1000)



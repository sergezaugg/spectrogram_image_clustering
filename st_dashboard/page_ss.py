

import streamlit as st
import os
from streamlit import session_state as ss
import gc
gc.collect()

st.text("Paths:")
st.write(ss['dapar']['feat_path'])
st.write(ss['dapar']['imgs_path'])

# st.divider()
st.divider()
st.text("Active data:")
st.write(ss['dapar']['dataset_name']  )
st.write(ss['dapar']['X'].shape           )
st.write(ss['dapar']['im_filenames'].shape )


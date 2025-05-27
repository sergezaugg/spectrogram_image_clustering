

import streamlit as st
import os
from streamlit import session_state as ss

st.write(ss['dapar']['feat_path'])
st.write(ss['dapar']['imgs_path'])


st.write(os.listdir(ss['dapar']['feat_path']))


st.write('dapar', ss['dapar'])
st.write('upar', ss['upar'])

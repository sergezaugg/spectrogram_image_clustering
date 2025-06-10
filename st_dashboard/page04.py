#--------------------             
# Author : Serge Zaugg
# Description : display tabular info on active data source 
#--------------------

import os
import streamlit as st
import pandas as pd
from streamlit import session_state as ss

# render page only id data is available 
if len(ss['dapar']['df_meta']) == 0 :
    st.text("First activate data (navigation bar left)")
else :
    # rename to keep code below readable 
    df_meta = ss['dapar']['df_meta']

    c00, c01  = st.columns([0.5, 0.2])
    with c00:
        with st.container(border=True) :   
            st.subheader('''Many thanks to the passionate people that shared recordings via xeno-canto.''')
            st.text('''For a detailed list of recordists see the field "rec" in the table below. Each files has a Creative Commons license, see field "lic" in the table below.''')
            st.dataframe(df_meta, hide_index = True, height=300)
            st.page_link("https://xeno-canto.org/", label=":gray[Link to xeno-canto web]")
        with st.container(border=True) :  
            c1, c2, c3, c4 = st.columns([0.2, 0.2, 0.2, 0.2])
            with c1:
                st.text("File primary species")
                dfsp = df_meta['full_spec_name'].value_counts().reset_index()
                dfsp = dfsp.sort_values(by = 'full_spec_name')
                st.dataframe(dfsp, use_container_width = True, hide_index = True)
            with c2:
                st.text("File country")
                dfcn = df_meta['cnt'].value_counts().reset_index()
                dfcn = dfcn.sort_values(by = 'cnt')
                st.dataframe(dfcn, use_container_width = True, hide_index = True)
            with c3:
                st.text("File durations")
                dfm = df_meta['length'].value_counts().reset_index()
                dfm = dfm.sort_values(by = 'length')
                st.dataframe(dfm, use_container_width = True, hide_index = True)
            with c4:
                st.text("File sampl. rates")
                dfs = df_meta['smp'].value_counts().reset_index()
                dfs = dfs.sort_values(by = 'smp')
                st.dataframe(dfs, use_container_width = True, hide_index = True)





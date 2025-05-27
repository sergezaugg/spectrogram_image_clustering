#--------------------             
# Author : Serge Zaugg
# Description : Info text
#--------------------

import streamlit as st
from streamlit import session_state as ss
import gc
gc.collect()

c00, c01  = st.columns([0.1, 0.18])
        
with c01:
    with st.container(border=True) : 
        
        st.markdown(''' 

            ### Data
         
            ### Feature extraction 
          
            ### Dimensionality reduction
          
            ### Clustering
                
            ### Clustering metrics
           
            ''')
        
   
 

      
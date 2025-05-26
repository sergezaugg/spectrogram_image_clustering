#--------------------             
# Author : Serge Zaugg
# Description : Info text
#--------------------

import streamlit as st
from streamlit import session_state as ss

c00, c01  = st.columns([0.1, 0.18])

# with c00:
#     with st.container(border=True) : 
#         # st.image(image = "pics/data_flow_chart_2.png", caption="Data flow chart", width=None, 
#         #          use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
         
with c01:
    with st.container(border=True) : 
        
        st.markdown(''' 

            ### Data
         
            ### Feature extraction 
          
            ### Dimensionality reduction
          
            ### Clustering
                
            ### Clustering metrics
           
            ''')
        
   
 

      
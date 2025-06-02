
#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import gc
from utils import display_mini_images_by_file

gc.collect()

display_mini_images_by_file(sel_imgs = np.array(ss['dapar']['saved_clusterd']),  num_cols = 10)





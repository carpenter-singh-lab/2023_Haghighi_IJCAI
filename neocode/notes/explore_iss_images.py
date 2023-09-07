"""Explore ISS images."""

#%%
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio

#%%
image_path = (
    Path(__file__)
    .parents[1]
    .joinpath("data/raw/iss/images/CP228A_Well1/Cycle03_T/Cycle03_T_Site_1.tiff")
)

#%%
image = skio.imread(image_path)
image
# %%
df = pd.DataFrame(image)
# %%
ndf = df.drop_duplicates()
ndf
# %%

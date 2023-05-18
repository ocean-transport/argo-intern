# Tracer stirring and variability in the Antarctic Circumpolar Current near the Southwest Indian Ridge
## Dhruv Balwada, Alison R. Gray, Lilian A. Dove, and Andrew F. Thompson
## Submitted to JGR:Oceans (in March 2023)

Analysis of tracer variability observed by sea gliders at scales smaller than ~100km.

A visual exploration of the glider can be done using the dashboard available [here](https://earthcube2021.github.io/ec21_book/notebooks/ec21_balwada_etal/README.html#). 


### Accessing the datasets
#### Glider data
The data for the two gliders (sg659 and sg660) can be downloaded at this [onedrive location](https://1drv.ms/u/s!AkfrsYYvKEb5jq8FmP18PQqkkwnStA?e=ATqUoL). At this location the data is stored in the format of two files (`*.eng and *.nc`) per profile. This data was reprocessed by [Geoff Shilling](https://apl.uw.edu/people/profile.php?last_name=Shilling&first_name=Geoff), who works with [Craig Lee](https://apl.uw.edu/people/profile.php?last_name=Lee&first_name=Craig) at the Applied Physics Lab, University of Washington.  

Some additonal QC was done on these data, and the resulting data is stored along in this repo in the `data` folder. All the analysis done in this study uses this post-processed data, and so most users should not need to download the data linked above. 

#### Roemmich-Gilson Argo Climatology
The RG Argo Climatology used to make figure 1 can be accessed from https://sio-argo.ucsd.edu/RG_Climatology.html.

#### Code order: 

1. Run the `QC_sg*.ipynb` files to do some additional QC on the data, which will be used for most of the analysis in the paper. (Generated the data files `sg_*_4m_binned.nc`)
2. Run Z_grid_to_isopycnal_grid.ipynb to do the interpolation from depth to isopycnal surfaces. (Generated the data files `sg_*_iso_grid.nc`)
3. Cut out RG Climatology for the SOGOS region using `select_RG_climatology_region.ipynb`.

Figure notebooks: 
- `sampling_resolution_plots.ipynb` - Generates figure to look at the horizontal and vertical sampling resolution of the gliders.
- 'Figure*.ipynb' - These notebooks generate the figures in the main text, and also some of the figures in the SI that are complementary to these main figures. 

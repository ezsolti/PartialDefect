The python modules and notebooks summarize work for assessing the partial defect identification capability of passive gamma spectroscopy.

Part I of this work investigates two things. 1, PCA of including several peaks from Cs134, Cs137 and Eu154 vs using only Eu154. 2, looks into the impact of calculating the inventory specific geometric efficiency.

Part II of this work is to be finalized. It will 1, extend the geometric efficiency calculation to include 3D effects. 2, add noise to the absolute peak area values 3, perform PCA and classification of up to 20 partial defect scenarios.

Dependencies:

- feign
- numpy
- pandas
- matplotlib
- sklearn
- xcom (not a python library)
- fuel library from http://dx.doi.org/10.17632/8z3smmw63p.1

Elements

- MVAfunctions.py: is a python module which contains several functions used in the notebooks
- PD-PartI.ipynb: the main notebook containing the creation of the peak counts-per-volume data and the analysis
- sample200.csv and outs/ folder: if someone does not want to redo the feign calculations, and the sampling of the fuel library, then there is a set of 200 samples, which can be used to play with the analysis part of the notebook.
- data/ folder contains attenuation coefficients for the feign calculations.
- detectorEfficiency/ folder contains a notebook and data to demonstrate how the detector efficiency curve was obtained and fitted.

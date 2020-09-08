The python modules and notebooks summarize work for assessing the partial defect identification capability of passive gamma spectroscopy.

Dependencies:

- feign
- numpy
- pandas
- matplotlib
- sklearn
- xcom (not a python library)

Besides some helper modules the main analysis is done in jupyter notebooks.

- Part I: investigates two things. 1, PCA of including several peaks from Cs134, Cs137 and Eu154 vs using only Eu154. 2, looks into the impact of calculating the inventory specific geometric efficiency.
- Part II: TODO. 1, will extend the geometric efficiency calculation to include 3D effects. 2, in order to get absolute peak area values and be able to add noise to  3, perform PCA and classification of up to 20 partial defect scenarios.

# CT Dataset

CT dataset can be downloaded [here](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144594475090).

Contact: [Paulo Ribeiro](mailto:paulo.ribeirodecarvalho@epfl.ch)

---

## Dataset

The dataset used in this project is from *Low Dose CT Grand Challenge*. The dataset comprises over fifteen thousand CT 
slices pair (low-full dose) from 10 patients, scanned under four different contexts: 1mm B30, 1mm D45, 3mm B30, and 
3mm D45.

## Data Re-structuration 

In the current folder, you find the `data_restructuration.py` file. This file will help you to easily restructure the 
data downloaded from *Low Dose CT Grand Challenge* to have the same structure as used during this project.

***Note:*** the results of the statistical analysis of the CT images can be loaded with the `ct_stat_results.npz` file.
This way, it is not mandatory to re-run the notebook to retrieve the distributions and other information.
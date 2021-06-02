no-covid-19-climate-silver-lining-in-the-us-power-sector
==============================

We calculate carbon emissions reductions due to decreases in U.S. electricity demand during COVID-19-related shutdowns. 

There are two modules to this repo: 
* **gaussian_processes** - This module corresponds to the analyses presented in Sections 2 and 3 of our paper
* **economic_modeling** - This module corresponds to the analyses presented in Section 4 of our paper.

If you use the code in this repo or otherwise would like to reference our study, please cite:

````
@article{luke2021no,
  title={No COVID-19 Climate Silver Lining in the US Power Sector},
  author={Luke, Max and Somani, Priyanshi and Cotterman, Turner and Suri, Dhruv and Lee, Stephen J},
  journal={Nature Communications},
  year={2021}
}
````

If you use the **gaussian_processes** module, know that it relies on the GPy library (https://github.com/SheffieldML/GPy). If you use this code, please also cite GPy:

````
@Misc{gpy2014,
  author =   {{GPy}},
  title =    {{GPy}: A Gaussian process framework in python},
  howpublished = {\url{http://github.com/SheffieldML/GPy}},
  year = {since 2012}
}
````


Python environment setup using Conda
------------
* Make sure you have the latest version of Anaconda installed that supports Python 3
* Create conda environment using provided yml file
    ~~~~
    conda env create -f environment.yml
    ~~~~

Configuration: using .env files for Python scripts
------------
* You will need to update the .env file for your specific environment:
    ~~~~
    PROJECT_ROOT=<local path to project root> 
    ~~~~

Running Gaussian Processes
------------
* To run the Gaussian Process (GP) analysis from data from pickle files, run:
~~~~
python gaussian_processes/src/models/run_gps_and_plot.py
~~~~
* To update the pickle files, run:
~~~~
python gaussian_processes/src/data/get_eia923_data.py
python gaussian_processes/src/data/get_eia923_emissions_by_fuel.py
~~~~

Running Economic Model
------------
* Please refer to the document: economic_modeling/README.md in this repo
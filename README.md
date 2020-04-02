<img src="notebooks/RUSH_full_color.jpg" align="center" width="300" height="70"/>

<font size="5" color='green'>Center for Quality, Safety & Value Analytics</font>

# SupplyDemand


## A python-based application for predicting changes in COVID-19 cases, hospital visits, admits, ICU needs, and protective equipment


<p style="text-align: justify;"><span style="color: rgb(0, 128, 0);">Developer</span></p>
<p style="text-align: justify;">Ken Locey, PhD Biology, Data Science Analyst</p>
<p style="text-align: justify;"><span style="color: rgb(0, 128, 0);">Site Architect and Administrator</span></p>
<p style="text-align: justify;">Jawad Khan, AVP, Advanced Analytics &amp; Knowledge Management</p>
<p style="text-align: justify;"><span style="color: rgb(0, 128, 0);">Center for Quality, Safety &amp; Value Analytics Leadership</span></p>
<p style="text-align: justify;">Thomas A. Webb, MBA, Associate Vice President</p>
<p style="text-align: justify;">Bala N. Hota, MD, MPH, Vice President, Chief Analytics Officer</p>

## Software used
### We tried to keep the required software to a minimum. Most are rather standard python libraries.

* python 3.7.4
* numpy 1.18.1
* voila 0.1.21 
	* The engine that converts jupyter notebooks to dashboard-like environments  https://voila.readthedocs.io/en/stable/index.html
* matplotlib 3.2.1
* pandas 1.0.3
* ipywidgets 7.5.1
* IPython 7.13.0
* scipy 1.4.1

## Websites

The source code in this repository is deployed to two different webistes:

1. Primary: Rush University Medical Center website <http://covid19forecast.rush.edu/> maintained by Jawad Khan
2. Testing: A heroku deployed app used for development and testing <https://rush-covid19.herokuapp.com/> maintained by Ken Locey and Tom Webb

## Contents

* **Procfile --** A file used by voila and heroku to set the environment. This file contains the following information and nothing else:
	```web: voila --port=$PORT --no-browser --enable_nbextensions=True --theme=light notebooks/dashboard.ipynb```

* **requirements.txt --** A file used by voila and heroku to determine the packages to download and install. The current contents of the file are:

	```
	voila
	voila-material
	matplotlib
	pandas
	ipywidgets
	numpy
	IPython
	scipy
	datetime
	```


* **runtime.txt --** A file needed to specify the python version used. The following is the only content of the file:

	```
	python-3.7.4. 
	
	```
	
* **notebooks --** A directory containing the following:

	* **app_class.py:** A highly commented python script containing functions used to construct the application's features (plots, tables, interactive widgets, etc.). This file is imported by `dashboard.ipynb`.
	
	* **model_fxns.py:** A highly commented python script containing fxns needed for modeling and statistical analyses. This file is imported by `app_class.py`
	
	* **get\_dataframe_dailyreports.py:** A python script that obtains and aggregates data from daily reports provided by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) <https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports>
	
	* **COVID-CASES-DF.txt:** The text file produced by `get_dataframe_dailyreports.py`. This file is used by `app_class.py` as the application's primary dataframe.
		
	* **dashboard.ipynb:** A jupyter notebook comprising the code that voila uses to generate a dashboard-like application. This file imports `app_class.py`.
	
	



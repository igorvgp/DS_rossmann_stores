<h1 align="center"> Creating a bot that predicts Rossmann future sales</h1>

<p align="center">
  <img src="https://github.com/igorvgp/rossman_stores/blob/main/Img/rossmann.jpg" alt="drawing" width="800"/>
</p>

*Obs: The business problem is fictitious, although both company and data are real.*

*The in-depth Python code explanation is available in [this](https://github.com/igorvgp/rossman_stores/blob/main/m10_v01_store_sales_prediction.ipynb) Jupyter Notebook.*

# 1. **Abstract**
<p align="justify"> This Data Science project was developed with Rossmann data available on [Kaggle](https://https://www.kaggle.com/) in order to predict sales of the next six weeks for each store and determine the best resource allocation for each store renovation.</p>
<p align="justify"> XGBoost machine learning model was trained to make the sales predictions, reaching a MAPE (mean percent error) of 14% and predicting a sales value of $283.7M in the following 6 weeks.</p>

<p align="justify"> The architecture of the project can is shown in the image below: </p>

<p align="center">
  <img src="https://github.com/igorvgp/rossman_stores/blob/main/Img/project_architecture.png" alt="drawing" width="800"/>
</p>

<p align="justify"> The solution was deployed at Heroku Cloud and the sales forecasts can be accessed through a Telegram bot available [here](https://t.me/rossmann_newapi_bot). </p>

<p align="center">
  <img src="https://github.com/igorvgp/rossman_stores/blob/main/Img/telegram_bot.jpeg" alt="drawing" width="800"/>
</p>

# 2. **Data Overview**
The data was collected from [Kaggle](https://www.kaggle.com/). This [dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data) contains historical sales data for 1,115 Rossmann stores. The initial features descriptions are available below:

| Feature | Definition |
|---|---|
| Id | an Id that represents a (Store, Date) duple within the dataset.|
| Store | a unique Id for each store.|
| Sales | the turnover for any given day.|
| DayOfWeek | day of week on which the sale was made (e.g. DayOfWeek=1 -> monday, DayOfWeek=2 -> tuesday, etc).|
| Date | date on which the sale was made.|
| Customers | the number of customers on a given day.|
| Open | an indicator for whether the store was open: 0 = closed, 1 = open.|
| StateHoliday | indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None.|
| SchoolHoliday  | indicates if the (Store, Date) was affected by the closure of public schools.|
| StoreType  | differentiates between 4 different store models: a, b, c, d.|
| Assortment | describes an assortment level: a = basic, b = extra, c = extended.|
| CompetitionDistance | distance in meters to the nearest competitor store.|
| CompetitionOpenSince(Month/Year)| gives the approximate year and month of the time the nearest competitor was opened.|
| Promo | indicates whether a store is running a promo on that day.|
| Promo2 | Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating.|
| Promo2Since(Year/Week)| describes the year and calendar week when the store started participating in Promo2.|
| PromoInterval | describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store.|

# 3. **Assumptions**
- Customers column was dropped, because for now there's no information about the amount of customers six weeks into the future. 
- The NaN's in CompetitionDistance were replaced by 3 times the maximum CompetitionDistance in the dataset, because the observations with NaN's are likely stores that are too far, which means there's no competition.
- Some new features were created in order to best describe the problem: 

| New Feature | Definition |
|---|---|
| day/week_of_year/year_week/month/year | day/week_of_year/year_week/month/year extracted from 'date' column.|
| day/day_of_week/week_of_year/month(sin/cos) | sin/cos component of each period, to capture their cyclical behavior.|
| competition_time_month| amount of months from competition start.|
| promo_time_week | time in weeks from when the promotion was active.|
| state_holiday(christmas/easter_holiday/public_holiday/regular_day)| indicates wheter the sale was made in christmas, easter, public holiday or regular day.|

# 4. **Solution Plan**
## 4.1. How was the problem solved?

<p align="justify"> To predict sales values for each store (six weeks in advance) a Machine Learning model was applied. To achieve that, the following steps were performed: </p>

- <b> Understanding the Business Problem </b> : Understanding the reasons why Rossmann's CEO was requiring that task, and plan the solution. 

- <b> Collecting Data </b>: Collecting Rossmann store and sales data from Kaggle.

- <b> Data Cleaning </b>: Renaming columns, changing data types and filling NaN's. 

- <b> Feature Engineering </b>: Creating new features from the original ones, so that those could be used in the ML model. 

- <p align="justify"> <b> Exploratory Data Analysis (EDA) </b>: Exploring the data in order to obtain business experience, look for useful business insights and find important features for the ML model. </a>. </p>

- <b> Data Preparation </b>: Applying Normalization and Rescaling Techniques in the data, as well as Enconding Methods and Response Variable Transformation.

- <b> Feature Selection </b>: Selecting the best features to use in the ML model by applying the <a href="https://www.section.io/engineering-education/getting-started-with-boruta-algorithm/">Boruta Algorithm</a>. 

- <p align="justify"> <b> Machine Learning Modeling </b>: Training Regression Algorithms with time series cross-validation. The best model was selected to be improved via Hyperparameter Tuning. </p>

- <b> Model Evaluation </b>: Evaluating the model using four metrics: MAE, MAPE and RMSE. 

- <b> Financial Results </b>: Translating the ML model's statistical performance to financial and business performance.

- <p align="justify"> <b> Model Deployment (Telegram Bot) </b>: Implementation of a Telegram Bot that will give you the prediction of any given available store number. This is the project's <b>Data Science Product</b>, and it can be accessed from anywhere. </p>
  
## 4.2. Tools and techniques used:
- Python 3.9.13, Pandas, Matplotlib, Seaborn and Sklearn.
- Jupyter Notebook and VSCode.
- Flask and Python API's.  
- Ngrok and Telegram Bot.
- Git and Github.
- Exploratory Data Analysis (EDA). 
- Techniques for Feature Selection.
- Regression Algorithms (Linear and Lasso Regression; Random Forest and XGBoost Regressors).
- Cross-Validation Methods, Hyperparameter Optimization and Algorithms Performance Metrics (RMSE, MAE and MAPE).

# 5. **Machine Learning Models**

<p align="justify"> This was the most fundamental part of this project, since it's in ML modeling where the sales predictions for each store can be made.  An average model was used as a baseline and four models were trained using time series cross-validation: </p>

- Linear Regression
- Lasso Regression (Regularized Linear Regression)
- Random Forest Regressor
- XGBoost Regressor

<p>The baseline model performance is displayed below: </p>

<div align="center">

| **Model Name** | **MAE** | **MAPE** | **RMSE** |
|:---:|:---:|:---:|:---:|
| Average Model | 1354.80 | 0.2064	 | 1835.135542 |

</div>

<p>The initial performance for all four algorithms are displayed below: </p>

<div align="center">

| **Model Name** | **MAE** | **MAPE** | **RMSE** |
|:---:|:---:|:---:|:---:|
| Random Forest Regressor | 1104.87 +/- 209.75 | 0.16 +/- 0.03	 | 1530.38 +/- 273.38 |
| XGBoost Regressor | 1179.33 +/- 111.96 | 0.17 +/- 0.01	 | 1639.3 +/- 148.34 |
| Linear Regression | 2079.0 +/- 280.91 | 0.3 +/- 0.01	 | 2955.05 +/- 426.56 |
| Lasso Regression | 2090.34 +/- 307.94 | 0.3 +/- 0.01 | 2995.12 +/- 458.89 |

</div>

<p align="justify"> Both Linear Regression and Lasso Regression have worst performances in comparison to the simple Average Model. This shows a nonlinear behavior in our dataset, hence the use of more complex models, such as Random Forest and XGBoost. </p>

<p align="justify"> <b> The XGBoost model was chosen for Hyperparameter Tuning. Even if Random Forest has the best performance if we look into the metrics, XGBoost would still be better to use, because it's much faster to train and tune </b>. </p>

<p>After tuning XGBoost's hyperparameters using Random Search the model performance has improved: </p>

<div align="center">
	
| **Model Name** | **MAE** | **MAPE** | **RMSE** |
|:---:|:---:|:---:|:---:|
| XGBoost Regressor | 949.881428	 | 0.143602 | 1336.919406 |


</div>

## 5.1. Brief Financial Results:

<p align="justify"> Below there are displayed two tables with brief financial results given by the XGBoost model. </p>

<p align="justify"> A couple interesting metrics to evaluate the financial performance of this solution is the MAE and MAPE. Below there's a table with a few stores metrics: </p>
<div align="center">

| **Store** | **Predictions (€)** | **Worst Scenario (€)** | **Best Scenario (€)** | **MAE (€)** | **MAPE** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  |	164,545.94 |	150,086.63 |	179,005.24 |	14,459.31 |	0.09 |
| 2  |	178,759.59 |	151,883.56 |	205,635.62 |	26,876.03 |	0.15 |
| 3  |	266,517.19 |	231,827.11 |	301,207.26 |	34,690.07 |	0.13 |
| 4  |	340,026.47 |	303,667.24 |	376,385.70 |	36,359.22 |	0.10 |
| 5  |	170,492.62 |	132,908.07 |	208,077.14 |	37,584.53 |	0.22 |
</div>

<p align="justify"> According to this model, the sales sum for all stores over the next six weeks is: </p>

<div align="center">

| **Scenario (€)** | **Total Sales of the Next 6 Weeks (€)** |
|:---:|:---:|
| Prediction  | $283,742,272.00 |
| Worst Scenario | $244,033,471.48 |
| Best Scenario | $323,451,121.16 |

</div>

# 6. **Model Deployment**

<p align="justify">  As previously mentioned, the complete financial results can be consulted by using the Telegram Bot. The idea behind this is to facilitate the access of any store sales prediction, as those can be checked from anywhere and from any electronic device, as long as internet connection is available.  
The bot will return you a sales prediction over the next six weeks for any available store, <b> all you have to do is send him the store number in this format "/store_number" (e.g. /12, /23, /41, etc) </b>. If a store number if non existent the message "Store not available" will be returned, and if you provide a text that isn't a number the bot will ask you to enter a valid store id. 

To link to chat with the Rossmann Bot is [![image](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/rossman_newapi_bot)

<i> Because the deployment was made in a free cloud (Render) it could take a few minutes for the bot to respond, <b> in the first request. </b> In the following requests it should respond instantly. </i>

</p>

# 7. **Conclusion**
In this project the main objective was accomplished:

 <p align="justify"> <b> A model that can provide good sales predictions for each store over the next six weeks was successfully trained and deployed in a Telegram Bot, which fulfilled CEO' s requirement, for now it's possible to determine the best resource allocation for each store renovation. </b></p>

 # Contact

- igorviniciusgpereira@gmail.com
- [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/igorvgpereira/)
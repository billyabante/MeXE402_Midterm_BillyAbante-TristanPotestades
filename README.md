# LINEAR REGRESSION

### Introduction

Linear regression is a statistical method used to model the relationship between a continuous dependent variable and one or more independent variables. This project aims to predict the number of daily or hourly bike rentals based on dependent variable (season, yr, mnth, day, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, and windspeed) that are influencing independent variable 'cnt' (bike rental counts per day or hour)  using linear regression.  
    
### Dataset Description

The Bike Sharing Dataset contains information about daily bike rentals in a specific region. It includes features like weather conditions, date information, and seasonal factors that might influence the number of bike rentals.

Bike-sharing rental process is highly correlated to the environmental and seasonal settings. For instance, weather conditions, precipitation, day of week, season, hour of the day, etc. can affect the rental behaviors. The core data set is related to the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare system, Washington D.C., USA which is publicly available in http://capitalbikeshare.com/system-data. They aggregated the data on two hourly and daily basis and then  extracted and added the corresponding weather and seasonal information. Weather information are extracted from http://www.freemeteo.com. 

##### Both hour.csv and day.csv have the following fields, except hr which is not available in day.csv
	- instant: record index
	- dteday : date
	- season : season (1:springer, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2011, 1:2012)
	- mnth : month ( 1 to 12)
	- hr : hour (0 to 23)
	- holiday : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : Normalized temperature in Celsius. The values are divided to 41 (max)
	- atemp: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
	- hum: Normalized humidity. The values are divided to 100 (max)
	- windspeed: Normalized wind speed. The values are divided to 67 (max)
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered

### Project Objectives

    1. Clean the data by handling missing values, identifying and treating outliers, and potentially normalizing features (if necessary).
    2. Build a linear regression model using the Scikit-learn library in Python to predict daily bike rentals.
    3. Evaluate the model's performance using metrics like R-squared and Adjusted R-squared.

### Methodology
#### 1. Data Preprocessing

Import Libraries: Necessary libraries like pandas and scikit-learn will be imported for data manipulation and modeling.
- import pandas as pd
- from sklearn.model_selection import train_test_split
- from sklearn.linear_model import LinearRegression

Load Data: Load the bike-sharing dataset using pandas.
   
Explore Data: Analyze basic statistics of the data to understand its characteristics and create new features or modify existing ones if it improves the model.
The 'dtype' column  is conveted into a datetime format for easier manipulation, then extracts only the day of the month from the date column because it has existing column for month and year. The 
unnecessary columns are removed, while the cnt column is renamed into total counts, and reorders the columns for better readability.
    
#### 2. Model Implementation

Split Data: Divide the data into training and testing sets. The training set will be used to build the model, and the testing set will be used to evaluate its performance on unseen data.
Define Model: Create a linear regression model instance using Scikit-learn's LinearRegression class.
Train Model: Fit the model on the training data.
Predict: Use the trained model to predict bike rentals on the testing data.

#### 3. Evaluation Metrics

Calculate R-squared and adjusted R-squared to assess how well the model explains the variance in the actual bike rentals.

### Results


    
### Discussion

    Reflect on the results and the limitations of the model.
    Discuss the significance of the features in predicting bike rentals.

# Logistic Regression

### Introduction

In our rapidly changing digital landscape, the way we collect and analyze data has become essential for driving innovation across various industries. With the sheer volume of data growing daily, the need for sophisticated tools that can interpret this information and provide actionable insights is more critical than ever. This is where Machine Learning (ML) comes into play, particularly through a technique called Regression Analysis, which helps us understand how different input factors (independent variables) influence specific outcomes (dependent variables). Among the various regression techniques available, I find Linear Regression and Logistic Regression particularly intriguing due to their diverse applications. Linear Regression is primarily used for predicting continuous outcomes, such as sales growth or changes in temperature. By identifying the best-fitting line through data points, this method helps us see trends and relationships clearly. For instance, using Multiple Linear Regression allows us to consider several factors, such as marketing spend and seasonality, to create a more comprehensive view of sales performance. In contrast, Logistic Regression is invaluable when dealing with categorical outcomes, especially binary results like whether a patient has a disease or not. This method utilizes a logistic function to determine probabilities for classification tasks, which is incredibly useful in fields like healthcare for disease prediction or finance for assessing credit risk. For example, a bank might use Logistic Regression to predict loan defaults based on various borrower characteristics. By integrating both Linear and Logistic Regression techniques, Machine Learning empowers us to extract valuable insights across diverse domains. This approach not only transforms raw data into actionable information but also helps us tackle real-world challenges more effectively.




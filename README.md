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

![image](https://github.com/user-attachments/assets/fbee4e02-8c8a-4eeb-aec4-c8e92c01a207)

Explore Data: Analyze basic statistics of the data to understand its characteristics and create new features or modify existing ones if it improves the model.
The 'dtype' column  is conveted into a datetime format for easier manipulation, then extracts only the day of the month from the date column because it has existing column for month and year. The 
unnecessary columns are removed, while the cnt column is renamed into total counts, and reorders the columns for better readability.

![image](https://github.com/user-attachments/assets/ef902978-bb21-4763-aadd-128d47885319)
    
#### 2. Model Implementation

Split Data: Divide the data into training and testing sets. The training set will be used to build the model, and the testing set will be used to evaluate its performance on unseen data.

Define Model: Create a linear regression model instance using Scikit-learn's LinearRegression class.

Train Model: Fit the model on the training data.

Predict: Use the trained model to predict bike rentals on the testing data.

![image](https://github.com/user-attachments/assets/9b972e9f-f701-4f23-81fc-b861f48002f7)

#### 3. Evaluation Metrics

Calculate R-squared and adjusted R-squared to assess how well the model explains the variance in the actual bike rentals.

![image](https://github.com/user-attachments/assets/0ec5efdf-0280-4e7d-85a0-fc2bf4f7019b)

### Results
In making the prediction of a single data point, the data (season, yr, mnth, day, holiday, weekday, workingday, weathersit, temp, atemp, hum, and windspeed) in row 10 were used. The result shows that during this condition the total bike rent can be 1333, while the data in row 10 with the same conditon has a total count of 1321 bike rented during that day.

    
### Discussion

The R-Squared of 0.8303 means that approximately 83% of the variation in bike rentals can be explained by the independent variables in the model. The adjusted R-Squared of 0.8151 suggests that, after accounting for the number of predictors, about 81.5% of the variation in bike rentals can be explained by the model. However, key factors influencing the count of bike rentals per day such as weather conditions, importatnt events, time-based factors, and holiday/working days. To improve the model utilize advanced techniques, incorporate external factors, and address importatnt events.

# Logistic Regression

### Introduction

In our rapidly changing digital landscape, the way we collect and analyze data has become essential for driving innovation across various industries. With the sheer volume of data growing daily, the need for sophisticated tools that can interpret this information and provide actionable insights is more critical than ever. This is where Machine Learning (ML) comes into play, particularly through a technique called Regression Analysis, which helps us understand how different input factors (independent variables) influence specific outcomes (dependent variables). Among the various regression techniques available, I find Linear Regression and Logistic Regression particularly intriguing due to their diverse applications. Linear Regression is primarily used for predicting continuous outcomes, such as sales growth or changes in temperature. By identifying the best-fitting line through data points, this method helps us see trends and relationships clearly. For instance, using Multiple Linear Regression allows us to consider several factors, such as marketing spend and seasonality, to create a more comprehensive view of sales performance. In contrast, Logistic Regression is invaluable when dealing with categorical outcomes, especially binary results like whether a patient has a disease or not. This method utilizes a logistic function to determine probabilities for classification tasks, which is incredibly useful in fields like healthcare for disease prediction or finance for assessing credit risk. For example, a bank might use Logistic Regression to predict loan defaults based on various borrower characteristics. By integrating both Linear and Logistic Regression techniques, Machine Learning empowers us to extract valuable insights across diverse domains. This approach not only transforms raw data into actionable information but also helps us tackle real-world challenges more effectively.

### Dataset Description

The BankNote Authentication dataset provides a comprehensive set of features derived from wavelet-transformed images of banknotes, specifically designed to help distinguish between authentic and counterfeit currency. This dataset is widely used in binary classification tasks and is particularly suited for predictive modeling, such as logistic regression. The data consists of several key attributes that reflect distinctive image properties crucial for accurate classification. These attributes include:

**Variance**: Measures the variation of the pixel intensity in the wavelet-transformed image.

**Skewness**: Indicates the asymmetry of the distribution in pixel intensity.

**Curtosis**: Describes the "peakedness" or sharpness of the intensity distribution.


**Entropy**: Represents the degree of randomness in the pixel intensity values.
The target variable, "Authenticity", labels the banknotes as either genuine or counterfeit.

This dataset is optimized for binary classification in predictive modeling, specifically with logistic regression, to analyze the key features influencing banknote authenticity.

To prepare the data for logistic regression, we encoded the target variable as follows:

**Authentic** - 1

**Counterfeit** - 0

### Project Objectives

1. To develop an accurate logistic regression model for BankNote authentication by analyzing features such as variance, skewness, curtosis, and entropy, identifying legitimate and counterfeit notes based on these distinct statistical attributes.

2. To evaluate model performance and effectiveness in real-time classification scenarios by measuring accuracy, precision, and other metrics, ensuring reliable detection of counterfeit notes to support banking and financial security applications.

### Methodology

## 1. Data Preprocessing 
   
Import Libraries: Load essential libraries for data handling and modeling, such as pandas and scikit-learn.

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

Load Data: Load the BankNote dataset using pandas.

![image](https://github.com/user-attachments/assets/d6c73886-f84e-4443-96ad-dbdb5dcfb009)

Explore Data: Analyze the dataset’s basic statistics to understand each feature (variance, skewness, curtosis, entropy). Assess feature distributions and check for any necessary modifications to enhance model performance. Adjustments may include scaling or transforming the features if required.

## 2. Model Implementation

Split Data: Divide the dataset into training and testing subsets. The training set will be used to build the logistic regression model, and the testing set will evaluate its performance on unseen data.

Define Model: Create an instance of the logistic regression model using Scikit-learn’s LogisticRegression class.

Train Model: Fit the logistic regression model on the training data.

Predict: Use the trained model to predict whether a banknote is legitimate or counterfeit on the testing data.

![image](https://github.com/user-attachments/assets/c8254097-e498-4873-a71f-4eb5c4da805d)

## 3. Evaluation Metrics

Calculate Accuracy and Confusion Matrix: Evaluate model performance by calculating the accuracy score and visualizing the confusion matrix. These metrics help assess the model's ability to classify banknotes as legitimate or counterfeit accurately.

![image](https://github.com/user-attachments/assets/56da867f-eb16-4cff-bd08-ecce956c16b5)


## Results
To predict the likelihood of a banknote being counterfeit, the model utilized the features from a specific data instance, including variance, skewness, curtosis, and entropy from row 10. The prediction indicated that under these specific conditions, the model suggests a classification of the banknote as legitimate, with a predicted probability of authenticity being 0.95. In comparison, the actual classification for the banknote in row 10 confirmed it as a genuine note, which supports the model's predictive capabilities.

## Discussion
An accuracy of 0.98 implies that approximately 98% of the variations in banknote authenticity can be attributed to the features included in the model. The precision of 0.95 indicates that a high proportion of predicted legitimate notes are indeed authentic, while a recall of 0.92 reveals that the model correctly identifies 92% of actual legitimate notes. This performance highlights the influence of critical factors, such as the statistical properties of the banknotes and their distinguishing features. To enhance the model further, it is recommended to explore more complex algorithms, integrate additional relevant features, and consider external influences that may affect banknote authenticity, such as production anomalies or regional variations in currency.


### Reference

## Bike sharing dataset

 https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset

## Bank note authentication

 https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data 







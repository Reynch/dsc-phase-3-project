# SyriaTel Churn Predictions

# Overview
![Towers](https://st.depositphotos.com/1968353/2536/i/450/depositphotos_25360787-stock-photo-communication-towers.jpg?v09-10-2023)

Our dataset consists of 3333 entries from customers with different phone plans, usage rates of different services, and customer service calls as well as if they churned or not.

# Bussiness Understanding
We would like to use a Logistic Regression model to help predict which customers will churn so that we can identify and try to prevent churn.


# Data

The data is from a Telecom company based in Syria. The data contains 20 feature columns and one target column - Churn.
The data is imbalanced marginally with 85.5% of the data being no churn and 14.5% being churn. The data could potentially use some balancing when we do our predictions but it may not be neccesary. With this data I created two different models that accomplish slightly different objectives.


# Modeling
![Cautious Model](./images/bestmodel1.png?)

My first model has a 90% accuracy and a 50% true positive rate. For example with a random sample of 100 Syria Tel users with 14 about to leave the company we would be able to find 7 cases of churn before they happen, misidentify 3 cases as churn when they would not actually leave the company and miss 7 cases of churn.


![Aggressive Model](./images/bestmodel2.png?)

My second model is a more aggressive model with an 86% accuracy and a 57% true positive rate. In this example we would be able to find 8 cases and only miss 6, however we would double the amount of individuals who were misidentified for churn.

# Evaluation

These are the factors that I found affected churn prediction the most. The international plan had the largest change with users 74% less likely to churn if they didn’t have the international plan and 250% more if they did. Next were customers with a higher amount of customer service calls as every customer service call they had increased their likelihood to churn. Finally, customers without a voicemail plan had a 67% increase to churn chance while customers who did had a 45% decrease to unsubscribe from the service. My recommendation would be to further investigate the large churn changers and see if there are any solutions, such as modifying the price of the international plan, incentivizing customers to sign up to the voicemail plan, and perhaps giving customers with frequent customer service calls priority to customer service agents.

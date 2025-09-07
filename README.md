Data_Science_Machine_Learning Project_Employee_Turnover_Analytics

# Employee Turnover Analytics

# Project Outline:
This project analyzes and predicts employee turnover at “Portobello Tech”, an app innovation company. The aim is to help the HR department to find trends that lead to employee attrition and make effective retention strategies with machine learning models.

Objective:
The main goal of this project is to make an end to end python ML pipeline for analyzing and predicting employee turnover at Portobello Tech. Using the give data having satisfaction level, last evaluation, project workload, working hours, tenure, promotions, department, and salary, the aim is to  find  whether employee leave the company with using  the appropriate  ways such as cluster employees who left based on work satisfaction,  class imbalance using SMOTE, and evaluating multiple classification models (Logistic Regression, Random Forest, Gradient Boosting) through cross-validation. By providing the best model and analyzing its predictions, the project aims to offer HR guidance on improving employee retention strategies.

#  Dataset
My project worked on the dataset “HR_comma_sep.csv” obtained from the MS AI  Machine Learning using Python with Python program with Simplilearn.

# Data analysis steps: 
All steps are mentioned into this “csv” file. 
Data_Science_Machine_Learning Project_Employee_Turnover_Analytics.csv

Steps:
1.	Check data quality 
2.	EDA: correlation heatmap, distributions, and bar plots.  
3.	Clustering: K-Means clustering of employees who left. 
4.	Data preprocessing: One-hot encoding used in categorical variables.  
5.	used “SMOTE” for balancing data
6.	Model training and evaluation (logistic regression, random forest classifier, gradient boosting classifier, 5-fold cross validation, roc-auc, and confusion matrix.  
7.	Retention strategy:  Predicted turnover probabilities and categorized employees into zones (safe zone (<20%), low-risk zone (20–60%), medium-risk zone (60–90%), high-risk zone (>90%).
   
# Conclusion
This project showed that employee turnover is modeled with machine learning techniques. The analysis presented main features influencing attrition, for instances poor satisfaction levels, long working hours, and lack of promotions. Clustering highlighted different groups of employees who are more likely to leave, while SMOTE balanced the dataset to improve model fairness. Among the tested models, ensemble methods (Random Forest and Gradient Boosting) show higher roc-auc and recall scores, making them more appropriate for turnover prediction. Finally, by categorizing employees into risk zones, the analysis presented clear strategies that can help HR address employee concerns, improve satisfaction, and then reduce attrition.

# Tools
•	Jupyter Notebook  
•	NumPy, Pandas, Matplotlib, Seaborn (Python)
•	Scikit-learn  
•	SMOTE  



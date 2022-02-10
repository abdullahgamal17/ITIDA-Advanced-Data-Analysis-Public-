# Project 3 "Communicate Data Findings"
This project was made as part of Udacity Advanced Data Analysis  
The data that I worked with is : Prosper Loan Data
# About Data Set
The data set consist of > 100,000 rows and 81 columns , showcasing different attributes gathered per each loan 
## Steps conducted in my analysis
 I extracted several categorical and non categorical data to use in data analysis then conducted on them univariate , bivariate and multivariate analysis to figure out what I can find out  

# Exploring the Data
## Univariate
I mainly divided them into two parts , categorical and non categorical , then analyzed all of them individually to understand a bit more about the data overall and to help me have an intuition which group of data can affect each other

## Bivariate
From the univariate analysis , I thought that {StatedMonthlyIncome, DebtToIncomeRatio, MonthlyLoanPayment, EmploymentStatusDuration} can have a relationship with { ProsperScore, CreditGrade, EmploymentStatus, LoanStatus} and showcased it using a map of boxplots

# Multivariate 
after conducting univariate and bivariate exploration , I thought I should figure out Employment Status , Monthly Income , Prosper Score Relationship , 
Income Range , BorrowerAPR , Prosper Score relationship , 
Credit Grade , Monthly income , Borrower APR , relationship,
I did that by a point plot , and Facet Grids

# Interesting Findings in Each Step
## Univariate
I found out that most credit grades fall in the middle categories (C , D and B)  
also found out that monthly income distribution is rightly skewed having a median of 5000 USD and if applied logarithmic transformation on it it turns into a normal distribution indicating low mean and high variance 

## Bivariate
Found out that there is a directly proportional relationship between prosper score and monthly income  
also found the same relationship between prosper score and monthly loan payment  
these results were observed by creating a grid of boxplots
## Multivariate
By creating a pointplot , I observed that their is an inversely proportional relationship between BorrowerAPR and prosper score  
and a direct relationship between monthly income and prosper score in case of employed and full time borrowers while that monthly income didn't improve the prosper score of Self employed borrowers

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)

df_ab_test = pd.read_csv('ab_data.csv')
df_countries = pd.read_csv('countries.csv')
print(df_ab_test.head())

print(f"The number of rows in the dataset is: {(df_ab_test.shape[0])}")
print(f"The number of unique users in the dataset is: {pd.unique(df_ab_test['user_id']).shape[0]}")
print(f"The number of converted users in the dataset is: {(df_ab_test['converted'].sum())}")

treatment = df_ab_test[df_ab_test['group']== 'treatment']
x = treatment[treatment['landing_page'] != 'new_page'].shape[0]

print(f'The number of times when the "group" is treatment but "landing_page" is not a new_page is {x} ')
missing = df_ab_test.isnull().sum().sum()
print(f"The number of missing values in the dataset is: {missing}")

# control group should have landing page = old_page , treatment group should have landing page = new_page
matching_rows_control = df_ab_test[(df_ab_test['group'] == 'control') & (df_ab_test['landing_page'] == 'old_page')]
matching_rows_treatment = df_ab_test[(df_ab_test['group'] == 'treatment') & (df_ab_test['landing_page'] == 'new_page')]
matching_rows = pd.concat([matching_rows_control, matching_rows_treatment])

print(f"The number of matching rows is: {matching_rows.shape[0]}")

#The number of unique users in the matching rows is:
unique_users = pd.unique(matching_rows['user_id']).shape[0]
print(f"The number of unique users in the matching rows is: {unique_users}")

#find the duplicate user in matching rows
duplicate_user = matching_rows[matching_rows.duplicated(subset='user_id', keep=False)]
print(duplicate_user['user_id'])

#display the rows of duplicates
print(duplicate_user)

#drop one duplicate from matching_users
matching_rows = matching_rows.drop(axis=0, index=duplicate_user.index[0])
print(matching_rows.shape[0])

#a)What is the probability of an individual converting regardless of the page they receive?
number_of_users = matching_rows.shape[0]
number_of_converted = matching_rows['converted'].sum()
probability_of_conversion = number_of_converted/number_of_users
print(f"The probability of an individual converting regardless of the page they receive is: {probability_of_conversion}")

#b)Given that an individual was in the control group, what is the probability they converted?
number_of_control_group_users = matching_rows[matching_rows['group'] == 'control'].shape[0]
number_of_converted_control_group = matching_rows[(matching_rows['group'] == 'control') & (matching_rows['converted'] == 1)].shape[0]
probability_of_conversion_control_group = number_of_converted_control_group/number_of_control_group_users
print(f"The probability of an individual being in the control group and converting is: {probability_of_conversion_control_group}")
#c)Given that an individual was in the treatment group, what is the probability they converted?
number_of_treatment_group_users = matching_rows[matching_rows['group'] == 'treatment'].shape[0]
number_of_converted_treatment_group = matching_rows[(matching_rows['group'] == 'treatment') & (matching_rows['converted'] == 1)].shape[0]
probability_of_conversion_treatment_group = number_of_converted_treatment_group/number_of_treatment_group_users
print(f"The probability of an individual being in the treatment group and converting is: {probability_of_conversion_treatment_group}")
#Calculate the difference in the probabilities absolute value
abs_difference = abs(probability_of_conversion_control_group - probability_of_conversion_treatment_group)
print(f"The absolute difference in the probabilities is: {abs_difference}")
#d)What is the probability that an individual received the new page
new_page_received = matching_rows[matching_rows['landing_page'] == 'new_page'].shape[0]
new_page_received_probability = new_page_received/number_of_users
print(f"The probability that an individual received the new page is: {new_page_received_probability}")

# e)Consider your results from parts (a) through (d) above, and explain below whether the new treatment group users lead to more conversions

new = matching_rows[matching_rows["group"] == "treatment"]
p_new = matching_rows[matching_rows["converted"] == 1].user_id.nunique()/matching_rows["user_id"].nunique()
print(p_new)

old = matching_rows[matching_rows["group"] == "control"]
p_old = matching_rows[matching_rows["converted"] == 1].user_id.nunique()/matching_rows["user_id"].nunique()
print(p_old)

num_new = new["user_id"].nunique()
print(num_new)

num_old = old["user_id"].nunique()
print(num_old)

# Simulate a Sample for the treatment Group
new_page_converted = np.random.choice(a=[0,1],size = num_new,p=[p_new,1-p_new])
old_page_converted = np.random.choice(a = [0,1],size = num_old ,p = [p_old,1-p_old])
converted_diff = new_page_converted.mean() - old_page_converted.mean()
p_diffs = []
s = matching_rows.shape[0]
sample = matching_rows.sample(s,replace=True)
for i in range(10000):
    
    new_page_converted = np.random.choice(a=[0,1],size = num_new,p=[p_new,1-p_new])
    old_page_converted = np.random.choice(a = [0,1],size = num_old ,p = [p_old,1-p_old])
    converted_diff = new_page_converted.mean() - old_page_converted.mean()
    p_diffs.append(converted_diff)

plt.hist(p_diffs,alpha = 0.5)
plt.xlabel("p'new - p'old")
plt.ylabel("Number of Occurences")
plt.title("Histogram represnting the sampling diffrence")
low  = np.percentile(p_diffs,0.5)
upper = np.percentile(p_diffs,99.5)
plt.axvline(abs_difference,color="black")
plt.axvline(low,color="red")
plt.axvline(upper,color="red")

count = 0
summ = 0

for i in range(len(p_diffs)):
    if(p_diffs[i] > abs_difference):
        count += 1
        summ += p_diffs[i]

x = count/len(p_diffs)

# number of conversions with the old_page
convert_old = matching_rows[(matching_rows['converted'] == 1) & (matching_rows['landing_page'] == 'old_page')].shape[0]

# number of conversions with the new_page
convert_new = matching_rows[(matching_rows['converted'] == 1) & (matching_rows['landing_page'] == 'new_page')].shape[0]

# number of individuals who were shown the old_page
n_old = matching_rows[matching_rows["landing_page"] == "old_page"].shape[0]

# number of individuals who received new_page
n_new = matching_rows[matching_rows["landing_page"] == "new_page"].shape[0]

print(convert_old)
print(convert_new)
print(n_old)
print(n_new)
    
import statsmodels.api as sm
# ToDo: Complete the sm.stats.proportions_ztest() method arguments
z_score, p_value = sm.stats.proportions_ztest([convert_old,convert_new],[n_old,n_new],value=None,alternative = "smaller",prop_var=False)
print(z_score, p_value)

matching_rows["intercept"] = 1
matching_rows = matching_rows.join(pd.get_dummies(matching_rows["landing_page"]))
matching_rows["ab_page"] = pd.get_dummies(matching_rows['group'])["treatment"]
logistic_model = sm.Logit(matching_rows["converted"],matching_rows[["intercept","ab_page"]])

answer = logistic_model.fit()
print(answer.summary())

df_countries = pd.read_csv("countries.csv")
print(df_countries.head())

df_merged = matching_rows.merge(df_countries, on ='user_id', how='left')
print(df_merged.head())

df_merged[['CA','UK','US']] = pd.get_dummies(df_merged['country'])
print(df_merged.head())

df_merged["intercept"] = 1
model = sm.Logit(df_merged['converted'],df_merged[["intercept","new_page","CA","UK","US"]])
answer = model.fit()
print(answer.summary())

df_merged["UK_new_page"] = df_merged["UK"]*df_merged["new_page"]
df_merged["CA_new_page"] = df_merged["CA"]*df_merged["new_page"]
df_merged["US_new_page"] = df_merged["US"]*df_merged["new_page"]
df_merged.head()
model = sm.Logit(df_merged['converted'],df_merged[["intercept","new_page","UK","US","UK_new_page","US_new_page"]])
answer = model.fit()
answer.summary2()

np.exp(answer.params)

print("Hello World")
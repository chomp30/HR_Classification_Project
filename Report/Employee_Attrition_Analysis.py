#Employee Attrition Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import altair as alt

df=pd.read_csv("D:\\Work\\Aline\\Projet_HR_classification\\HR_Classification_Project\\Files\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
df_clean=pd.read_csv("D:\\Work\\Aline\\Projet_HR_classification\\HR_Classification_Project\\Files\\df_clean.csv")
df_clean_num=pd.read_csv("D:\\Work\\Aline\\Projet_HR_classification\\HR_Classification_Project\\Files\\df_clean_num.csv")

st.title("Attrition in HR : Analysis and Classification")
st.image("D:\\Work\\Aline\\Projet_HR_classification\\HR_Classification_Project\\Report\\Attrition_image.jpg")
st.sidebar.title("Summary")
pages=["Exploration","Data Visualization","Modelization", "Conclusion"]
page=st.sidebar.radio("Go to", pages)
st.write("The gradual erosion of motivation, the loss of a sense of belonging to a corporate culture, the gradual disengagement of employees, sometimes leading to voluntary departures, are all factors in the attrition phenomenon.")
st.write("Since Covid, companies from all sectors are more and more looking into reasons of attrition and how to prevent it before it happens. With this work, my aim is to find the reasons of attrition and predict if an employee is at risk in order to help managers and companies to find solutions.")


if page==pages[0]:
  st.write("### Exploration")
  st.write("My dataset is composed of the following DataFrame :")
  st.dataframe(df.head(15))
  st.write(df.shape)
  if st.checkbox("Showcase the missing values"):
    st.dataframe(df.isna().sum())
  st.write("In this DataFrame, we have 35 columns describing various elements about the employees, either personal info such as: age, gender, marital status, education level... and professional background with info regarding their current status within the company : job title, their department, their traveling habits and their incomes.\n And we have one line per employee so a total of 1470 lines.")
  st.write("Some variables are already encoded like:")
  st.write(" - Education with the values : 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'")
  st.write(" - Environment Satisfaction means the following: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'")
  st.write(" - Job Involvement means: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'")
  st.write(" - Job Satisfaction means: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'")
  st.write(" - Performance Rating is displayed like : 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'")
  st.write(" - Relationship Satisfaction with the manager is showcased : 1 'Low' 2 'Medium' 3 'High' 4 'Very High'")
  st.write(" - And Work Life Balance measure is displayed as 1 'Bad' 2 'Good' 3 'Better' 4 'Best'")
  st.write("The goal is to simplify the DataFrame, so I am keeping the Monthly Income but I am droping the Monthly and Daily Rates which are not key for my analysis. I am also dropping the Standard Hours column as they are the same of all employees observed : 80.")
  st.write("The Over18 variable is not important (all employees have over 18 years old) so I am dropping it as well as the Employee Number.")
  st.write("To finish, my target variable is Attrition, in order to make it ready for machine-learning classification model, I will encode it so I have now Yes=1 and No=0.")
  st.write("So the DataFrame is now cleaned and ready for Dataviz and analytics and looks like this:")
  st.dataframe(df_clean.head(15))
  st.write(df_clean.shape)
  st.write("As Attrition is my target variable, let's look at the repartition of the values it contains. We can see it is uneven, so this is something I will need to look for when it comes to modelization:")
  st.write(df_clean["Attrition"].value_counts())
  st.write("Let's now look at the other key categories: Gender, Marital Status, Business Travel and Job Role")
  st.write(df_clean["Gender"].value_counts(), df_clean["MaritalStatus"].value_counts(), df_clean["BusinessTravel"].value_counts(), df_clean["JobRole"].value_counts())

if page==pages[1]:
  st.write("### Data Visualization")
  st.write("For the analysis and then prediction, my target variable will be Attrition. The visualization of data as well as statistics analysis will be in regards of this target. The goal is to determine if one of more variables are influencing my target and if I can draw a portrait of an employee at risk of attrition.")
  st.write("\n")
  st.write("First, I will dive into the distribution of the Age and Monthly Income variables.")

  fig3= sns.displot(df_clean["Age"], kde=True,rug=True,bins=15,color="green")
  plt.title=("Repartition of the variable : Age")
  st.pyplot(fig3)

  fig4= sns.displot(df_clean["MonthlyIncome"], kde=True,bins=15, color="pink")
  plt.title=("Repartition of the variable : Monthly Income")
  st.pyplot(fig4)

  st.write("Now, let's see how the Monthly income is acting towards other variables. Especially within the departments, then, following gender and years at company, to finish with marital status. Monthly Income is clearly an important variable to understand the attrition or the absence of it. We often believe that if the pay is higher, attrition is lower, we'll see if this applies here.")
  df_satisfaction=df_clean.groupby("Department").agg({"MonthlyIncome":"mean"})
  st.write(df_satisfaction.sort_values(by="MonthlyIncome", ascending=False))
  st.write("Here, the mean of monthly income per department is quite even, Sales department has the highest mean, followed by HR and R&D department.")

  fig=px.area(df_clean, x="MonthlyIncome", y="YearsAtCompany", color="Gender",title='Monthly Income per years at company and gender')
  fig.update_traces(textposition="bottom center")
  st.plotly_chart(fig, theme=None, use_container_width=True)
  st.write("This graph is very interesting as it showcases that the highest incomes are clearly for men but are not particularly linked to the years within the company.")

  st.write("Let's map the composition of the workforce in this company by looking at its seniority and current position:")
  chart_data = df_clean
  c = (
   alt.Chart(chart_data)
   .mark_circle()
   .encode(x="TotalWorkingYears", y="YearsInCurrentRole", size="YearsWithCurrManager", color="JobRole", tooltip=["TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager"])
  )

  st.altair_chart(c, use_container_width=True)

  fig2 = sns.catplot(x="Age",y="Gender",kind="box",col="Attrition", data=df_clean)
  plt.figure(figsize=(10,10))
  st.pyplot(fig2)

  fig7 = sns.catplot(x="MaritalStatus", y="MonthlyIncome", kind="box", col="Attrition", data=df_clean)
  st.pyplot(fig7)


  sales_attrition=df_clean.loc[(df_clean['Department']=="Sales") & (df_clean['Attrition']==1)]
  st.write("Let's see the number of employees concerned by attrition within the sales department:")
  st.write(len(sales_attrition))

  job_satis=df_clean.groupby(['JobRole']).agg({"MonthlyIncome":"mean",
                                            "JobSatisfaction":"count"})
  job_satis.sort_values(by="JobSatisfaction", ascending=False)

  job_attrition=df_clean.groupby(['JobRole']).agg({"Attrition":"count"})
  job_gender=df_clean.groupby(['JobRole']).agg({"Gender":"count"})

  fig5 = go.Figure()
  fig5.add_traces([go.Bar(name='Monthly Income',
                       x=job_satis.index,
                       y=job_satis['MonthlyIncome'], marker_color="#87a96b")])

  fig5.update_layout(title="Monthly income per job role")
  st.plotly_chart(fig5, theme=None, use_container_width=True)

  st.write("Job Role repartition following highest job satisfaction score:")
  fig6= plt.figure(figsize=(10,10))
  plt.pie(x=job_satis['JobSatisfaction'].sort_values(ascending=False), labels=["Sales Executive", "Research Scientist", "Lab Technician", "Manufacturing Director", "Healthcare Rep", "Manager", "Sales Rep", "Research Director", "HR"],
       colors=["#b69872","#ff7f50","#5f646d","#d3dacf","#ffe085","#62daff","#ffc03e","#fdc0cc","#e6e6fa"], explode=[0.1,0.1,0.1,0,0,0,0,0,0],
        autopct=lambda x:round(x,2).astype(str)+"%", pctdistance=0.7, labeldistance=1.1)
  plt.title=("Job Role with the highest job satisfaction score")
  plt.legend(bbox_to_anchor=(1,1.1), loc="upper left")
  st.pyplot(fig6)

  year_attrition=df_clean.groupby(['Attrition']).agg({"TotalWorkingYears":"mean",
                                                   'YearsSinceLastPromotion':"mean",
                                                   "YearsAtCompany":"mean",
                                                   "YearsWithCurrManager":"mean",
                                                   "YearsInCurrentRole":"mean",
                                                   "Age":"mean"})

  st.dataframe(year_attrition)
  st.write("In this DataFrame, I have done the average of each variable, so I can show the fact, on average younger workers, with less seniority are more at risk of attrition. An important point to note : the number of years since last promotion are pretty close wether it is an attrition risk or not, so it means that recognition seems not to be a key factor for attrition. Another important insight, the number of years with the current manager is higher when there are no attrition, it means that managers are a key variable for an employee to leave its job or not.")
#Here we can see that the employees who are more at risk to leave the company are the younger ones, they have been in the role
#since less time and within the company less years than the rest (by the way, an attrition after 5 years in the company
#is a very good score compared to other tech companies today). We can note that it is not promotion that leads to attrition
#as the mean of years since last promotion are very close but the gap is important when it comes to manager. Employees at risk
#of attrition have spent half less time with their current manager than the employees who are staying
  cor = df_clean_num.corr() 
  fig8, ax = plt.subplots(figsize = (15,15))
  sns.heatmap(cor, annot = True, ax = ax, cmap = "coolwarm")
  st.pyplot(fig8)


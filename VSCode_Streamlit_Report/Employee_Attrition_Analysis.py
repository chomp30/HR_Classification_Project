#Employee Attrition Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import altair as alt
import joblib
from sklearn.metrics import classification_report

df=pd.read_csv("VSCode_Streamlit_Report/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df_clean=pd.read_csv("VSCode_Streamlit_Report/df_clean.csv")
df_clean_num=pd.read_csv("VSCode_Streamlit_Report/df_clean_num.csv")
df_X_clean=pd.read_csv("VSCode_Streamlit_Report/df_X_clean.csv")
tsne_acp_df_clean=pd.read_csv("VSCode_Streamlit_Report/tsne_acp_df_clean.csv")

st.title("Attrition in HR : Analysis and Classification")
st.image("VSCode_Streamlit_Report/Attrition_image.jpg")

st.sidebar.title("Summary")
pages=["Exploration","Data Visualization","Modelization", "Conclusion"]
page=st.sidebar.radio("Go to", pages)



if page==pages[0]:
  st.write("The gradual erosion of motivation, the loss of a sense of belonging to a corporate culture, the gradual disengagement of employees, sometimes leading to voluntary departures, are all factors in the attrition phenomenon.")
  st.write("Since Covid, companies from all sectors are more and more looking into reasons of attrition and how to prevent it before it happens. With this work, my aim is to find the reasons of attrition and predict if an employee is at risk in order to help managers and companies to find solutions.")
  st.header('Exploration', divider='red')
  st.write("The data I will work with are taken from Kaggle, my dataset is composed of the following DataFrame :")
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
  st.header('Data Visualization', divider='red')
  st.write("For the analysis and then prediction, my target variable will be Attrition. The visualization of data as well as statistics analysis will be in regards of this target. The goal is to determine if one of more variables are influencing my target and if I can draw a portrait of an employee at risk of attrition.")
  st.write("\n")
  st.write("First, I will dive into the distribution of the Age and Monthly Income variables.")

  fig3= sns.displot(df_clean["Age"], kde=True,rug=True,bins=15,color="green")
  plt.title=("Repartition of the variable : Age")
  st.pyplot(fig3)

  fig4= sns.displot(df_clean["MonthlyIncome"], kde=True,bins=15, color="pink")
  plt.title=("Repartition of the variable : Monthly Income")
  st.pyplot(fig4)

  st.write("Both Age and Monthly Income columns are following a normal law graph. For the Age variable, we can see a peak around 33 years old and for the Monthly Income around 3000$. The Monthly Income is less well distributed as we can see a set of extreme numbers.")

  st.write("Now, let's see how the Monthly income is acting towards other variables. Especially within the departments, then, following gender and years at company, to finish with marital status. Monthly Income is clearly an important variable to understand the attrition or the absence of it. We often believe that if the pay is higher, attrition is lower, we'll see if this applies here.")
  df_satisfaction=df_clean.groupby("Department").agg({"MonthlyIncome":"mean"})
  st.write(df_satisfaction.sort_values(by="MonthlyIncome", ascending=False))
  st.write("Here, the average of monthly income per department is quite even, Sales department has the highest mean, followed by HR and R&D department.")

  fig=px.area(df_clean, x="MonthlyIncome", y="YearsAtCompany", color="Gender",title='Monthly Income per years at company and gender')
  fig.update_traces(textposition="bottom center")
  st.plotly_chart(fig, theme=None, use_container_width=True)
  st.write("This graph is very interesting as it showcases that the highest incomes are for men (and we saw that there are more men than women in the company) but are not totally linked to the years within the company. Some employees who arrived less that a year in the company can get top salaries.")

  st.write("Let's map the composition of the workforce in this company by looking at its seniority and current position:")
  chart_data = df_clean
  c = (
   alt.Chart(chart_data)
   .mark_circle()
   .encode(x="TotalWorkingYears", y="YearsInCurrentRole", size="YearsWithCurrManager", color="JobRole", tooltip=["TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager"])
  )

  st.altair_chart(c, use_container_width=True)
  st.write("At this company, the employees have mostly 10-15 years of experience overall, with some outliers with 40+ years of experience. The Lab Technicians and Sales Exec are the 2 positions less represented after 20 years of experience, whereas Managers are the opposite")

  fig2 = sns.catplot(x="Age",y="Gender",kind="box",col="Attrition", data=df_clean)
  plt.figure(figsize=(10,10))
  st.pyplot(fig2)
  st.write("Here, we can see that the risk of attrition for women is on the average at 30 years old, whether for men it is around 32/33 years old. We can notice some outliers for women leaving the company at age 50 and plus")

  fig7 = sns.catplot(x="MaritalStatus", y="MonthlyIncome", kind="box", col="Attrition", data=df_clean)
  st.pyplot(fig7)
  st.write("When looking at marital status by monthly income and separated by attrition, clearly the employee staying at the company have a higher monthly incomes for the 3 different marital status. The lowest average income for employee with attrition are for the single and divorced persons")


  sales_attrition=df_clean.loc[(df_clean['Department']=="Sales") & (df_clean['Attrition']==1)]
  st.write("Let's see the number of employees concerned by attrition within the sales department:")
  st.write(len(sales_attrition))

  hr_attrition=df_clean.loc[(df_clean['Department']=="Human Resources") & (df_clean['Attrition']==1)]
  st.write("Let's see the number of employees concerned by attrition within the HR department:")
  st.write(len(hr_attrition))

  rd_attrition=df_clean.loc[(df_clean['Department']=="Research & Development") & (df_clean['Attrition']==1)]
  st.write("Let's see the number of employees concerned by attrition within the R&D department:")
  st.write(len(rd_attrition))

  st.write("R&D department is the most concerned by attrition, followed by Sales and HR. It makes sense as the R&D department is the largest one.")

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
  st.write("Sales Executive, Research Scientist and Lab Technician are the 3 job roles with the higher satisfaction score. HR are the ones with the lowest grade for job satisfaction.")

  fig9=sns.catplot(x="Age", y="JobRole", hue="Attrition", kind="bar", data=df_clean)
  plt.figure(figsize=(15,15))
  st.pyplot(fig9)

  year_attrition=df_clean.groupby(['Attrition']).agg({"TotalWorkingYears":"mean",
                                                   'YearsSinceLastPromotion':"mean",
                                                   "YearsAtCompany":"mean",
                                                   "YearsWithCurrManager":"mean",
                                                   "YearsInCurrentRole":"mean",
                                                   "Age":"mean"})

  st.dataframe(year_attrition)
  st.write("In this DataFrame, I have done the average of each variable, so I can show the fact that, on average, younger workers with less seniority are more at risk of attrition. An important point to note : the number of years since last promotion are pretty close wether it is an attrition risk or not, so it means that recognition seems not to be a key factor for attrition. Another important insight, the number of years with the current manager is higher when there are no attrition, it means that managers are a key variable for an employee to decide to leave his/her job or not.")
#Here we can see that the employees who are more at risk to leave the company are the younger ones, they have been in the role
#since less time and within the company less years than the rest (by the way, an attrition after 5 years in the company
#is a very good score compared to other tech companies today). We can note that it is not promotion that leads to attrition
#as the mean of years since last promotion are very close but the gap is important when it comes to manager. Employees at risk
#of attrition have spent half less time with their current manager than the employees who are staying
  cor = df_clean_num.corr() 
  fig8, ax = plt.subplots(figsize = (15,15))
  sns.heatmap(cor, annot = True, ax = ax, cmap = "coolwarm")
  st.pyplot(fig8)
  st.write("The heatmap is very unclear to read because no variables seem to be correlated to my target variable : attrition. The variable that is a bit more linked to Attrition is OverTime. Let's deep dive into dimension reduction in order to get the most important variable before processing a machine-learning model for classification.")

if page==pages[2]:
  st.header('Modelization', divider='red')
  st.write("For this work, I am facing a classification prediction with 2 classes : 0 = no attrition, 1=attrition. As explained before, my classes are unbalanced, the class 1 being undersampled. To compare the predictions, I will create a Decision Tree Classifier model, a Balanced Random Forest Classifier model to boost the prediction on class 1 and a Logistic Regression.")
  st.write("First, let's reduce the dimension of the dataset with PCA and T-SNE and visualize which characteristics are important to determine attrition. The data have been encoded with a Standard Scaler as well as a One Hot Encoder.")
  from sklearn.preprocessing import StandardScaler
  scaler=StandardScaler()
  Z=scaler.fit_transform(df_X_clean)
  from sklearn.decomposition import PCA
  pca=PCA()
  coord=pca.fit_transform(Z)

  fig10= plt.figure(figsize=(10,10))
  plt.plot(np.arange(1,48), pca.explained_variance_)
  plt.xlabel('Nombre de facteurs')
  plt.ylabel("Valeurs propres")
  st.pyplot(fig10)
  st.write("The PCA explained variance shows that we have around 8 factors that are important to determine attrition.")
  
  st.write("Here we can understand which variables are important for the attrition target.")
  size=len(df_X_clean.columns)
  racine_valeurs_propres=np.sqrt(pca.explained_variance_)
  corvar=np.zeros((size,size))
  for k in range(size):
      corvar[:,k]=pca.components_[:,k]*racine_valeurs_propres[k]
    
  fig,axes=plt.subplots(figsize=(20,20))
  axes.set_xlim(-1,1)
  axes.set_ylim(-1,1)

  for j in range(size):
      plt.annotate(df_X_clean.columns[j],(corvar[j,0]*0.8,corvar[j,1]*0.8), color='red')
      plt.arrow(0,0,corvar[j,0]*0.6, corvar[j,1]*0.6, alpha=0.5, head_width=0.03, color="b")
    
  plt.plot([-1,1],[0,0], color="silver", linestyle="-", linewidth=1)
  plt.plot([0,0],[-1,1], color="silver", linestyle='-', linewidth=1)

  cercle=plt.Circle((0,0),1, color="green",fill=False)
  axes.add_artist(cercle)
  plt.xlabel("PC 1")
  plt.ylabel("PC 2")
  st.pyplot(fig)
  st.write("The attrition is linked to PC1. ")
  st.write("Let's combine the T-SNE with PCA and visualize the positions of the different employees at risk of attrition or not:")
  
  fig11=plt.figure(figsize=(10,10))
  sns.scatterplot(x="Axe 1", y="Axe 2", hue="Target", data=tsne_acp_df_clean)
  st.pyplot(fig11)

  st.write("So, the people at risk of attrition are mostly in the bottom left area of the graphic and at the top. In that cases the variables regarding the number of years with current manager, years at company, gender as well as monthly income & age are key variables here.")
  st.write("We still have some outliers of attrition in other areas but the point of focus should be the variables quoted above.")

  # Liste déroulante
  no_model = "Select a model"
  model_1 = "Decision Tree Classifier - Max Depth 4"
  model_2 = "Balanced Random Forest"
  model_3 = "Logistic Regression"
  model_options = [no_model, model_1, model_2, model_3]
  selected_model = st.selectbox('Selection du modèle:', model_options)

  # Si sélection d'un modèle
  if selected_model != no_model:
        
      # Variables
      if selected_model == model_1:
          model_type = "Decision Tree Classifier - Max Depth 4"
          model_depth = "4"
          model_loaded = joblib.load("VSCode_Streamlit_Report/DecisionTreeClassifier_attrition.joblib")
          model_loaded2= joblib.load("VSCode_Streamlit_Report/DecisionTreeClassifier_pred.joblib")
      if selected_model == model_2:
          model_type = "Balanced Random Forest"
          model_depth = "4"
          model_loaded = joblib.load("VSCode_Streamlit_Report/BalancedRandomForestClassifier.joblib")
          model_loaded2=joblib.load("VSCode_Streamlit_Report/BalancedRandomForest_pred.joblib")
      if selected_model == model_3:
          model_type = "Logistic Regression"
          model_depth = "Max"
          model_loaded = joblib.load("VSCode_Streamlit_Report/LogisticRegression.joblib")
          model_loaded2=joblib.load("VSCode_Streamlit_Report/LogisticRegression_pred.joblib")
      # Présentation du Modèle
      st.write('### Model Selected')
      st.write('Model Type:', model_type)
      st.write('Model Depth:', model_depth)

      # Checkbox
      st.write("### Options:")
      FeatImp_button_status = st.checkbox("Showcase the Feature Importances")
      Xtest_button_status = st.checkbox("Charge a test sample and make a prediction")
      PersPred_button_status = st.checkbox("Create a personalized prediction")

       # Feature Importances Matrix
      if FeatImp_button_status == True:
          st.write('### Feature Importances Matrix')
          X_train_columns = joblib.load("VSCode_Streamlit_Report/X_train_columns")
          feature_importances = pd.DataFrame({'Variable' : X_train_columns, 'Importance' : model_loaded.feature_importances_}).sort_values('Importance', ascending = False)
          st.dataframe(feature_importances[feature_importances['Importance'] > 0.02])
      # Chargement du jeu de test
      if Xtest_button_status == True:
          X_test = joblib.load("VSCode_Streamlit_Report/X_test.joblib")
          # X_test = pd.read_csv("Streamlit/vgsales_RandomForestReg_Xtest.csv", index_col = 0)
          y_test = joblib.load("VSCode_Streamlit_Report/y_test.joblib")
          X_test_decoded = pd.read_csv("VSCode_Streamlit_Report/X_test.csv", index_col = 0)
          st.write('### Presentation of the test sample')
          st.write('Number of employees listed:', X_test.shape[0])
          st.write("Extract from the encoded dataset:")
          st.dataframe(X_test.head(5))

          # Prédiction sur jeu de test
          pred_button_status = st.button("Make a prediction")

          if pred_button_status == True:
              st.write("Accuracy Score", model_loaded.score(X_test, y_test))
              y_pred = model_loaded.predict(X_test)
              X_test_decoded['Attrition - Predicted'] = y_pred
              X_test_decoded['Attrition - Real'] = y_test
                                   
              st.write("##### Classification report")
              st.dataframe(classification_report(y_test, y_pred, output_dict=True))
              st.write("The classification report showcase that the Balanced Random Forest Classifier is the best prediction model with a 75% accuracy score. When can see that the recall score for the class 1 is around 60% which means that the ratio to predict the attrition is rather positive even if the precision can be low for the same class.")
              st.write("The Decision Tree Classifier has a 84% accuracy score but is clearly overfitting with the training sample and also the recall score for the class 1 is around 25%, showcasing that this model predicts very well the class 0 but can't really predict attrition.")
              st.write("The logistic regression model is showcasing a good accuracy score as well, around 85% but the class 1 still remains not well predicted with only 5% for the recall score.")

        # Faire une prédiction personnalisée
      if PersPred_button_status == True:

        # Définition des valeurs
          st.write("The values have been taken from the PCA and T-SNE dimension reduction as well as the feature importances section, so we can see a prediction from a smaller set of columns with the key variables identified.")
          MonthlyIncome_options = [Income for Income in range(1000,20000)]
          Gender_options = [0, 1]
          JobLevel_options = [1,2,3,4,5]
          TotalWorkingYears_options = [Year for Year in range(0,40)]
          YearsWithManager_options = [Year for Year in range(0,17)]
          Age_options = [Age for Age in range(18, 65)]

          input_MonthlyIncome = st.select_slider("Monthly Income", MonthlyIncome_options)
          input_Gender= st.selectbox("Gender", Gender_options)
          input_JobLevel = st.selectbox("Job Level", JobLevel_options)
          input_TotalWorkingYears = st.select_slider("Total Working Years", TotalWorkingYears_options)
          input_YearsWithManager = st.select_slider("Years with Current Manager", YearsWithManager_options)
          input_Age = st.select_slider("Age", Age_options)

# Faire une prédiction
          perspred_button_status = st.button("Faire une prédiction")

          if perspred_button_status == True:
              df_X_clean_pred=pd.DataFrame({"Age":input_Age,
                                            "Gender":input_Gender,
                                             "JobLevel": input_JobLevel,
                                             "TotalWorkingYears":input_TotalWorkingYears,
                                             "MonthlyIncome":input_MonthlyIncome,
                                              "YearsWithCurrManager":input_YearsWithManager}, index=[0])
              
              st.write("##### Summary")
              st.dataframe(df_X_clean_pred)

            # Prédiction
              y_perso_pred = int(np.round(model_loaded2.predict(df_X_clean_pred))[0])
              y_perso_pred = "{:,.0f}".format(y_perso_pred)
              st.metric("Predicted Attrition", y_perso_pred)

if page==pages[3]:
  st.header('Conclusion', divider='red')
  st.write("To conclude this analysis on attrition, those are my insights to the HR Execs, based on the study :")
  st.write("**Managers are key** : A part of attrition is due to management issues, regular check-ups must be mandatory as well as manager training to make sure they understand the needs and feedbacks from their team.")
  st.write("**Income is a topic** : Employees need to discuss their salaries, especially in the tech ecosystem where there's a lot of competition, you need to make sure that you are paying your employees within the right pay range.")
  st.write("**Gender must be a priority**: Attrition is also due to gender inequality, especially in tech environment which has few women. The discrepancy in pay is a key point for attrition.")
  st.write("To go further, in order to maximise the machine-learning model to predict attrition, it would be interesting to deep-dive into a specific ecosystem, to benchmark other companies from the same sector as well as finding other data on the level of stress, work/life balance and company's values.")
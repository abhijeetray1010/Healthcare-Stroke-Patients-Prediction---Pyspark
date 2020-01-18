# Databricks notebook source
# MAGIC %md ##Healthcare Stroke Patients Prediction - Pyspark
# MAGIC 
# MAGIC Team members - Abhijeet Ray, Deepak Rao , AKhil Menon, Kushagra Sen

# COMMAND ----------

# MAGIC %md ## Objective 
# MAGIC * To forecast whether a patient can have stroke or not using the given information of patients. It is a classification problem, where we will try to predict the probability of an observation belonging to a category (in our case probability of having a stroke)
# MAGIC 
# MAGIC * Here we have clinical measurements (e.g. Hypertension, heart_disease, age, smoking_status) for a number of patients, as well as information about whether each patient has had a stroke. In practice, we are developing our model to accurately predict stroke risk for future patients based on their clinical measurements.

# COMMAND ----------

# MAGIC %md ### Understand stroke
# MAGIC #### What is a stroke
# MAGIC 
# MAGIC * A stroke is a “brain attack”. It can happen to anyone at any time. It occurs when blood flow to an area of brain is cut off. When this happens, brain cells are deprived of oxygen and begin to die. When brain cells die during a stroke, abilities controlled by that area of the brain such as memory and muscle control are lost.
# MAGIC 
# MAGIC #### Stroke are the world’s biggest killers
# MAGIC 
# MAGIC * 800,000 strokes per year in the US
# MAGIC * Fitfh leading cause of death in the US
# MAGIC * Leading cause of adult disability in the US
# MAGIC * 80% are preventable

# COMMAND ----------

# MAGIC %md #### Understanding stroke attributes in the dataset
# MAGIC 
# MAGIC * This dataset contains clinical measurements of 43401 patients. Description of this dataset can be viewed on the Kaggle website, where the data was obtained (https://www.kaggle.com/asaumya/healthcare-dataset-stroke-data).
# MAGIC 
# MAGIC ##### Data Description
# MAGIC * ID: Patient's ID, probably irrelevant unless to avoid duplicates
# MAGIC 
# MAGIC * GENDER - SEX (male ; female)
# MAGIC 
# MAGIC * AGE - age in years
# MAGIC 
# MAGIC * HYPERTENSION: (0- No ; 1- Yes) Hypertension is another name for high blood pressure, Can lead to severe complications and increases the risk of heart disease, stroke, and death. Normal blood pressure is 120 over 80 mm of mercury (mmHg), but hypertension is higher than 130 over 80 mmHg. (Source: https://www.medicalnewstoday.com/articles/150109.php)
# MAGIC 
# MAGIC * HEART_DISEASE: (0- No ; 1- Yes), could cause increase stroke risks.
# MAGIC      
# MAGIC * EVER_MARRIED: (YES or NO) Lifestyle factor. Is this relevant? -We will need to test.
# MAGIC 
# MAGIC * TYPE_OF_WORK: Did the person worked as a Government servant or in a Private organization or was Self employed? Lifestyle factor. Is this relevant? -We will need to test.
# MAGIC 
# MAGIC * RESIDENCE: Home location (Rural or Urban), Lifestyle factor. Is this relevant? -We will need to test.
# MAGIC 
# MAGIC * AVG_GLUCOSE: Average Glucose level in the person, could be relevant.
# MAGIC 
# MAGIC * BMI: Body Mass Index. Could be relevant, to test.
# MAGIC 
# MAGIC      *An index for assessing overweight and underweight, obtained by dividing body weight in kg by height in m^2
# MAGIC      
# MAGIC      *A measure of 25 or more is considered overweight.
# MAGIC      
# MAGIC * Smoking_Status : Smoking habbits (Smoking, Formerly smoked, occassional smoker), its a Lifestyle factor.
# MAGIC 
# MAGIC ##### Predictor Field
# MAGIC * Stroke: Did the person had stroke ? (0-No ; 1-Yes)

# COMMAND ----------

# MAGIC %md ### Importing necessary modules

# COMMAND ----------

from pyspark.sql import SparkSession
import pyspark.sql as sparksql
import pandas as pd
import numpy as np
#Data visualisation libraries 
import matplotlib.pyplot as plt   
import seaborn as sns
#importing ml features
from pyspark.ml.feature import StringIndexer,VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline,Model
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

# MAGIC %sh
# MAGIC wget -O train https://www.dropbox.com/s/p3sy3sxj71gxjtx/train_2v.csv?dl=0

# COMMAND ----------

# MAGIC %md ### Loading the dataset

# COMMAND ----------

train = spark.read.csv(path='file:///databricks/driver/train', inferSchema=True,header=True)

# COMMAND ----------

# MAGIC %md ### Exploring the data

# COMMAND ----------

train.dtypes

# COMMAND ----------

train.columns

# COMMAND ----------

train.printSchema()

# COMMAND ----------

train.show()

# COMMAND ----------

# MAGIC %md ### Visualizing the Data

# COMMAND ----------

pdata = train.toPandas()
g = sns.pairplot(pdata, size=1.2)
display(g.fig)

# COMMAND ----------

fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
sns.heatmap(pdata.corr(),annot=True ,cmap='magma').set_title('Correlation Factors Heat Map', color='black', size='20')
display(fig)

# COMMAND ----------

# MAGIC %md * It is clearly evident that Age and heart disease are the factors that are highly correlated with Stroke, followed by avg_glucose_level, hypertension and then bmi

# COMMAND ----------

# MAGIC %md #### Training feature analysis

# COMMAND ----------

# create DataFrame as a temporary view for SQL queries
train.createOrReplaceTempView('table')

# COMMAND ----------

# MAGIC %md #### Work type vs stroke (influence of work type on getting stroke)

# COMMAND ----------

# sql query to find the number of people in specific work_type who have had stroke and not
spark.sql("SELECT work_type, COUNT(work_type) as work_type_count \
          FROM table WHERE stroke == 1 \
          GROUP BY work_type \
          ORDER BY COUNT(work_type) DESC").show()

# COMMAND ----------

# MAGIC %md * The most affected work_type persons are private followed by self-employed.

# COMMAND ----------

# MAGIC %md #### Gender vs Stroke (is strokes related to gender ? )

# COMMAND ----------

spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT COUNT(gender) FROM table WHERE gender == 'Male') as    percentage \
          FROM table WHERE stroke== 1 AND gender = 'Male' \
          GROUP BY gender").show()
spark.sql("SELECT gender, COUNT(gender) as gender_count, COUNT(gender)*100/(SELECT COUNT(gender) FROM table WHERE gender == 'Female') as percentage \
          FROM table WHERE stroke== 1 AND gender = 'Female' \
          GROUP BY gender").show()

# COMMAND ----------

# MAGIC %md * 1.68% male and almost 2% male had stroke.

# COMMAND ----------

# MAGIC %md #### Age vs Stroke (Now we will see influence of age on stroke)

# COMMAND ----------

spark.sql("SELECT COUNT(age)*100/(SELECT COUNT(age) FROM table WHERE stroke ==1) as percentage \
          FROM table \
          WHERE stroke == 1 AND age<50").show()
spark.sql("SELECT COUNT(age)*100/(SELECT COUNT(age) FROM table WHERE stroke ==1) as percentage \
          FROM table \
          WHERE stroke == 1 AND age>=50").show()

# COMMAND ----------

# MAGIC %md * It is clearly evident that older age have a strong correlation with Strokes, as 91.5% stroke had occured for people who are more than 50 years old and only 8% stroke for people whose age is less than 50

# COMMAND ----------

# MAGIC %md ### Cleaning up training data

# COMMAND ----------

train.describe().show()

# COMMAND ----------

# MAGIC %md d
# MAGIC  * Here we see that the count for each of the columns is 43400 except "smoking status" and "bmi". This indicates there are few missing values in "smoking_status" and "bmi"
# MAGIC  * Also there are few categorical data (gender, ever_married, work_type, Residence_type, smoking_status) which we need to covert using one hot encoding

# COMMAND ----------

# fill in missing values for column "smoking status"
# As "smoking status" column is categorical data, we will add one data type "No Info" for the missing one

train_f = train.na.fill('No Info', subset=['smoking_status'])

# COMMAND ----------

# fill in missing values for column "bmi" 
# as column "bmi" is numerical data , we will fill the missing values with mean

from pyspark.sql.functions import mean
mean = train_f.select(mean(train_f['bmi'])).collect()
mean_bmi = mean[0][0]
train_f = train_f.na.fill(mean_bmi,['bmi'])

# COMMAND ----------

train_f.describe().show()

# COMMAND ----------

# MAGIC %md d
# MAGIC  * Here we see that the count for each of the columns now is 43400. This indicates the missing values in "smoking_status" and "bmi" has been handled

# COMMAND ----------

# MAGIC %md ### Data Modelling

# COMMAND ----------

# MAGIC %md * Now, Lets work on categorical columns.
# MAGIC * Perfroming "StringIndexer -> OneHotEncoder -> VectorAssembler"

# COMMAND ----------

# indexing all categorical columns in the dataset

indexer1 = StringIndexer(inputCol="gender", outputCol="genderIndex")
indexer2 = StringIndexer(inputCol="ever_married", outputCol="ever_marriedIndex")
indexer3 = StringIndexer(inputCol="work_type", outputCol="work_typeIndex")
indexer4 = StringIndexer(inputCol="Residence_type", outputCol="Residence_typeIndex")
indexer5 = StringIndexer(inputCol="smoking_status", outputCol="smoking_statusIndex")

# COMMAND ----------

# Doing one hot encoding of indexed data

from pyspark.ml.feature import OneHotEncoderEstimator
encoder = OneHotEncoderEstimator(inputCols=["genderIndex","ever_marriedIndex","work_typeIndex","Residence_typeIndex","smoking_statusIndex"],
                                 outputCols=["genderVec","ever_marriedVec","work_typeVec","Residence_typeVec","smoking_statusVec"])

# COMMAND ----------

#The next step is to create an assembler, that combines a given list of columns into a single vector column to train ML model. We will use the vector columns, that we got after one_hot_encoding.
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=[
 'genderVec',
 'age',
 'hypertension',
 'heart_disease',
 'ever_marriedVec',
 'work_typeVec',
 'Residence_typeVec',
 'avg_glucose_level',
 'bmi',
 'smoking_statusVec'] , outputCol='features')

# COMMAND ----------

# MAGIC %md * We have complex task that contains bunch of stages, these bunch of satges needs to be performed to process data. To wrap all of that Spark ML represents such a workflow as a Pipeline, which consists of a sequence of PipelineStages to be run in a specific order.

# COMMAND ----------

from pyspark.ml import Pipeline
# Pipeline basic to be shared across model fitting and testing
pipeline = Pipeline(stages=[]) # Must initialize with empty list!

# COMMAND ----------

basePipeline = [indexer1, indexer2, indexer3, indexer4, indexer5, encoder, assembler]

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, NaiveBayes, LogisticRegression, DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator

dtc = DecisionTreeClassifier(labelCol='stroke',featuresCol='features')
pl_dtc=basePipeline+[dtc]
pg_dtc = ParamGridBuilder()\
        .baseOn({pipeline.stages: pl_dtc})\
        .build()

rf = RandomForestClassifier(featuresCol="features", labelCol="stroke",numTrees=25)
pl_rf = basePipeline + [rf]
pg_rf = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_rf})\
      .build()

nb = NaiveBayes(labelCol='stroke',featuresCol='features')
pl_nb = basePipeline + [nb]
pg_nb = ParamGridBuilder()\
.baseOn({pipeline.stages: pl_nb})\
.addGrid(nb.smoothing,[0.4,1.0])\
.build()

paramGrid = pg_dtc + pg_rf + pg_nb


# COMMAND ----------

# MAGIC %md ###Data Transformation

# COMMAND ----------

splitted_data = train_f.randomSplit([0.7,0.3], 24)
train_data = splitted_data[0]
test_data = splitted_data[1]
print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))

# COMMAND ----------

cv = CrossValidator()\
    .setEstimator(pipeline)\
    .setEvaluator(MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy"))\
    .setEstimatorParamMaps(paramGrid)\
    .setNumFolds(3) 

# COMMAND ----------

fittedModel1=cv.fit(train_data)

# COMMAND ----------

train_data.show()

# COMMAND ----------

# MAGIC %md ###The Best Model

# COMMAND ----------

import numpy as np
# BinaryClassificationEvaluator defaults to ROC AUC, so higher is better
# http://gim.unmc.edu/dxtests/roc3.htm
fittedModel1.getEstimatorParamMaps()[ np.argmax(fittedModel1.avgMetrics) ]

# COMMAND ----------

# MAGIC %md * As is clearly evident, RandomeForestCLassifier is coming out to be the best model

# COMMAND ----------

# MAGIC %md ###The Worst Model

# COMMAND ----------

fittedModel1.getEstimatorParamMaps()[ np.argmin(fittedModel1.avgMetrics) ]

# COMMAND ----------

# MAGIC %md * NaiveBayes is worst model

# COMMAND ----------

import re
def paramGrid_model_name(model):
  params = [v for v in model.values() if type(v) is not list]
  name = [v[-1] for v in model.values() if type(v) is list][0]
  name = re.match(r'([a-zA-Z]*)', str(name)).groups()[0]
  return "{}{}".format(name,params)

# Resulting metric and model description
# get the measure from the CrossValidator, cvModel.avgMetrics
# get the model name & params from the paramGrid
# put them together here:
kmeans_measures = zip(fittedModel1.avgMetrics, [paramGrid_model_name(m) for m in paramGrid])
metrics,model_names = zip(*kmeans_measures)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.clf() # clear figure
fig = plt.figure( figsize=(5, 5))
plt.style.use('fivethirtyeight')
axis = fig.add_axes([0.1, 0.3, 0.8, 0.6])
# plot the metrics as Y
#plt.plot(range(len(model_names)),metrics)
plt.bar(range(len(model_names)),metrics)
# plot the model name & param as X labels
plt.xticks(range(len(model_names)), model_names, rotation=70, fontsize=6)
plt.yticks(fontsize=6)
#plt.xlabel('model',fontsize=8)
plt.ylabel('ROC AUC (better is greater)',fontsize=8)
plt.title('Model evaluations')
display(plt.show())

# COMMAND ----------

# MAGIC %md ### Predictions

# COMMAND ----------

predictions=fittedModel1.transform(test_data)

# COMMAND ----------

## Make predictions on test documents. 
# CrossValidator.fit() is in cvModel, which is the best model found (rfModel).
predictions.select("prediction", "stroke").show(5)

# COMMAND ----------

# MAGIC %md ### Model Evaluation

# COMMAND ----------

# Select (prediction, true label) and compute test error

acc_evaluator = MulticlassClassificationEvaluator(labelCol="stroke", predictionCol="prediction", metricName="accuracy")
rfc_acc = acc_evaluator.evaluate(predictions)
print('A Random Forest Classifier had an accuracy of: {0:2.2f}%'.format(rfc_acc*100))

# COMMAND ----------

# MAGIC %md #### So, the best model which is the RandomForestClassifier gives us the accuracy of 98.30% which is very high, so our model's accuracy seems to be great

# COMMAND ----------

# MAGIC %md ####Lets view result of the best model

# COMMAND ----------

correct = predictions.where("(stroke = prediction)").count()
incorrect = predictions.where("(stroke != prediction)").count()

resultDF = sqlContext.createDataFrame([['correct', correct], ['incorrect', incorrect]], ['metric', 'value'])
display(resultDF)

# COMMAND ----------

# MAGIC %md ## Result Visualization

# COMMAND ----------

correct = predictions.where("(stroke = prediction)").count()
incorrect = predictions.where("(stroke != prediction)").count()

resultDF = sqlContext.createDataFrame([['correct', correct], ['incorrect', incorrect]], ['metric', 'value'])
display(resultDF)

# COMMAND ----------

# MAGIC %md #### As is evident from the above visualization, our model predicted very high correct classifications over incorrect classifications which clearly indicates model is performing good.

# COMMAND ----------

# MAGIC %md ####Confusion Matrix

# COMMAND ----------

counts = [predictions.where('stroke=1').count(), predictions.where('prediction=1').count(),
          predictions.where('stroke=0').count(), predictions.where('prediction=0').count()]
names = ['actual 1', 'predicted 1', 'actual 0', 'predicted 0']
display(sqlContext.createDataFrame(zip(names,counts),['Measure','Value']))

# COMMAND ----------

counts = [predictions.where('stroke=1').count(), predictions.where('prediction=1').count(),
          predictions.where('stroke=0').count(), predictions.where('prediction=0').count()]
names = ['actual 1', 'predicted 1', 'actual 0', 'predicted 0']
display(sqlContext.createDataFrame(zip(names,counts),['Measure','Value']))

# COMMAND ----------



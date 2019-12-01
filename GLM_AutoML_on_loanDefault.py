import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\sbortnem\\Desktop\\Fall 2019\\Anomaly Detection - APAN 5420\\Assignment 10\\XYZloan_default_selected_vars.csv")

df.head()

df.shape

df = df.drop(['Unnamed: 0','Unnamed: 0.1','id'], axis =1)

train, test = train_test_split(df, test_size=0.4, random_state=42)


# **Viewing and understanding NAs**

def NA_check (df):
    percent_NA = df.isnull().sum() * 100 / len(df)
    NA_df = pd.DataFrame({'percent_NA': percent_NA})
    return NA_df[NA_df.percent_NA > 0]

NA_check(train).head()
NA_check(test).head()


# There are features with 99%+ missing values. These would not be useful in data analysis, so I am removing them.
def percent_99_nas (df):
    p_na = df.isnull().sum() * 100 / len(df)
    cols_p_nas = p_na[p_na > 99].index
    df.drop(cols_p_nas, axis=1, inplace=True)

percent_99_nas(test)
percent_99_nas(train)

test.shape
train.shape


# Additionally, there are columns that only contain a single value. These should also be removed, as they are not useful in seeing patterns within the data.
def one_value_columns (df):
    n = df.apply(pd.Series.nunique)
    cols = n[n == 1].index
    df = df.drop(cols, axis=1)
    return df

train = one_value_columns(train)
test = one_value_columns(test)

train.shape
test.shape


# **Feature engineering and binning** 
# The purpose of the analysis is to find patterns. Because of this, I am transforming the date variable to create a month variable. All data occurs in 2017, so I cannot create a year variable.

train['AP005'] = pd.to_datetime(train['AP005'])
train['AP005_month'] = pd.DatetimeIndex(train['AP005']).month

train[['AP005','AP005_month']].head()


test['AP005'] = pd.to_datetime(test['AP005'])
test['AP005_month'] = pd.DatetimeIndex(test['AP005']).month

test[['AP005','AP005_month']].head()


# Next I will bin continuous variables to again ensure pattern forming. 

def cols_for_binning (df):
    bin_cols = df.columns[df.isin([-99,-999,0,np.nan]).any()] 
    bin_cols = bin_cols.drop(['loan_default','AP009'])
    return bin_cols

def bin_cols (df):
    cols_to_bin = cols_for_binning(df)
    for col in cols_to_bin:
        name = col +'_bin'
        df[name] = pd.qcut(df[col],q=[0, .2, .4, .6, .8, 1], duplicates = 'drop')
    
    return df


train = bin_cols(train)
train.head()

test = bin_cols(test)
test.head()


def replace_values (df):
    df_new = df[df.columns.drop(list(df.filter(regex='_bin')))] #removing bin cols first
    bin_df = cols_for_binning(df_new)
    
    #creating specific bins for -999, -99, 0, and No Data, as they represent differing values
    for col in bin_df:
        name = col +'_bin'
        df[name] = df[name].cat.add_categories(new_categories = ['-99','-999','0','No Data'])
        df.loc[df[col] == -99, name] = '-99'
        df.loc[df[col] == -999, name] = '-999'
        df.loc[df[col] == 0, name] = '0'
        df.loc[df[col].isnull(), name] = 'No Data'
    return df


train = replace_values(train)
train.head()

test = replace_values(test)
test.head()



train_without_bins = train[train.columns.drop(list(train.filter(regex='_bin')))]
cols_remove_train = cols_for_binning(train_without_bins)
train = train.drop(cols_remove_test, axis = 1)

test_without_bins = test[test.columns.drop(list(test.filter(regex='_bin')))]
cols_remove_test = cols_for_binning(test_without_bins)
test = test.drop(cols_remove_train, axis = 1)


train.shape
test.shape


# Creating age bins to introduce repeatability.

train['AP001_bins'] = pd.cut(train['AP001'], bins=[20,24,29,34,39,49,56],labels=['20 to 24','25 to 29','30 to 34','35 to 39','40 to 49','50 and above'])
train[['AP001','AP001_bins']].head()


test['AP001_bins'] = pd.cut(test['AP001'], bins=[20,24,29,34,39,49,56],labels=['20 to 24','25 to 29','30 to 34','35 to 39','40 to 49','50 and above'])
test[['AP001','AP001_bins']].head()


# **Creating variable lists**
vars = pd.DataFrame(train.dtypes).reset_index()
vars.columns = ['name','dtype'] 
vars['var_source'] = vars['name'].str[:2]
vars.var_source.value_counts()

MB_vars = list(vars[vars.var_source =='MB']['name'])
AP_vars = list(vars[(vars.var_source =='AP') & (vars['name']!='AP004') & (vars['name']!='AP001')]['name'])
TD_vars = list(vars[vars.var_source =='TD']['name'])
CR_vars = list(vars[vars.var_source =='CR']['name'])
PA_vars = list(vars[vars.var_source =='PA']['name'])
CD_vars = list(vars[vars.var_source =='CD']['name'])


train['loan_default'].value_counts(dropna=False)


# ## GLM
import h2o
h2o.init()

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator


target = 'loan_default'


train_s = train.sample(frac=0.1, random_state=1)
test_s = test.sample(frac=0.1, random_state=1)
train_hex = h2o.H2OFrame(train_s)
test_hex = h2o.H2OFrame(test_s)


# ### GLM Model 1
pred_vars = CR_vars + TD_vars + AP_vars + MB_vars + CD_vars + PA_vars

glm1 = H2OGeneralizedLinearEstimator(family= "binomial", lambda_ = 0, nfolds = 10)
glm1.train(pred_vars,target,training_frame=train_hex)

predictions = glm1.predict(test_hex)['p1']
test_scores = test_hex['loan_default'].cbind(predictions).as_data_frame()
test_scores.head(15)


def createGains(model):
    predictions = model.predict(test_hex)['p1']
    test_scores = test_hex['loan_default'].cbind(predictions).as_data_frame()

    #sort on prediction (descending), add id, and decile for groups containing 1/10 of datapoints
    test_scores = test_scores.sort_values(by='p1',ascending=False)
    test_scores['row_id'] = range(0,0+len(test_scores))
    test_scores['decile'] = ( test_scores['row_id'] / (len(test_scores)/10) ).astype(int)
    #see count by decile
    test_scores.loc[test_scores['decile'] == 10]=9
    test_scores['decile'].value_counts()

    #create gains table
    gains = test_scores.groupby('decile')['loan_default'].agg(['count','sum'])
    gains.columns = ['count','actual']
    gains

    #add features to gains table
    gains['non_actual'] = gains['count'] - gains['actual']
    gains['cum_count'] = gains['count'].cumsum()
    gains['cum_actual'] = gains['actual'].cumsum()
    gains['cum_non_actual'] = gains['non_actual'].cumsum()
    gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
    gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
    gains['if_random'] = np.max(gains['cum_actual']) /10 
    gains['if_random'] = gains['if_random'].cumsum()
    gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
    gains['K_S'] = np.abs( gains['percent_cum_actual'] -  gains['percent_cum_non_actual'] ) * 100
    gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
    gains = pd.DataFrame(gains)
    return(gains)

createGains(glm1)


def ROC_AUC(my_result,df,target):
    from sklearn.metrics import roc_curve,auc
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    # ROC
    y_actual = df[target].as_data_frame()
    y_pred = my_result.predict(df)['p1'].as_data_frame()
    fpr = list()
    tpr = list()
    roc_auc = list()
    fpr,tpr,_ = roc_curve(y_actual,y_pred)
    roc_auc = auc(fpr,tpr)
    
    # Precision-Recall
    average_precision = average_precision_score(y_actual,y_pred)

    print('')
    print('   * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate')
    print('')
    print('	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy')
    print('')
    print('   * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)')
    print('')
    
    # plotting
    plt.figure(figsize=(10,4))

    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve (aare=%0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=3,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: AUC={0:0.4f}'.format(roc_auc))
    plt.legend(loc='lower right')

    # Precision-Recall
    plt.subplot(1,2,2)
    precision,recall,_ = precision_recall_curve(y_actual,y_pred)
    plt.step(recall,precision,color='b',alpha=0.2,where='post')
    plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title('Precision-Recall curve: PR={0:0.4f}'.format(average_precision))
    plt.show()


ROC_AUC(glm1,test_hex,'loan_default')

# **Coefficients table**

coefs = glm1._model_json['output']['coefficients_table'].as_data_frame()
coefs = pd.DataFrame(coefs)
coefs.reindex(coefs.standardized_coefficients.abs().sort_values(ascending = False).index).head(10)




# ### GLM Model 2
glm2 = H2OGeneralizedLinearEstimator(family= "binomial", lambda_ = 0.003, nfolds = 10)
glm2.train(pred_vars,target,training_frame=train_hex)

predictions2 = glm2.predict(test_hex)['p1']
test_scores2 = test_hex['loan_default'].cbind(predictions2).as_data_frame()
test_scores2.head(15)

createGains(glm2)

ROC_AUC(glm2,test_hex,'loan_default')

# **Coefficients table**
coefs = glm2._model_json['output']['coefficients_table'].as_data_frame()
coefs = pd.DataFrame(coefs)
coefs.reindex(coefs.standardized_coefficients.abs().sort_values(ascending = False).index).head(10)




# ### GLM Model 3
glm3 = H2OGeneralizedLinearEstimator(family= "binomial", lambda_ = 0.0015, nfolds = 10, alpha = 1)
glm3.train(pred_vars,target,training_frame=train_hex)

predictions3 = glm3.predict(test_hex)['p1']
test_scores3 = test_hex['loan_default'].cbind(predictions3).as_data_frame()
test_scores3.head(15)

createGains(glm3)

ROC_AUC(glm3,test_hex,'loan_default')

# **Coefficients table**
coefs = glm3._model_json['output']['coefficients_table'].as_data_frame()
coefs = pd.DataFrame(coefs)
coefs.reindex(coefs.standardized_coefficients.abs().sort_values(ascending = False).index).head(10)






# ## AutoML

from h2o.automl import H2OAutoML
aml1 = H2OAutoML(max_runtime_secs = 90, max_models=20, seed=1)
aml1.train(pred_vars,target,training_frame=train_hex)

aml1.leaderboard

pred = aml1.predict(test_hex)
pred.head()

performance = aml1.leader.model_performance(test_hex)
performance


def createGainsAml(model):
    predictions = model.predict(test_hex)
    test_scores = test_hex['loan_default'].cbind(predictions).as_data_frame()

    #sort on prediction (descending), add id, and decile for groups containing 1/10 of datapoints
    test_scores = test_scores.sort_values(by='predict',ascending=False)
    test_scores['row_id'] = range(0,0+len(test_scores))
    test_scores['decile'] = ( test_scores['row_id'] / (len(test_scores)/10) ).astype(int)
    #see count by decile
    test_scores.loc[test_scores['decile'] == 10]=9
    test_scores['decile'].value_counts()

    #create gains table
    gains = test_scores.groupby('decile')['loan_default'].agg(['count','sum'])
    gains.columns = ['count','actual']
    gains

    #add features to gains table
    gains['non_actual'] = gains['count'] - gains['actual']
    gains['cum_count'] = gains['count'].cumsum()
    gains['cum_actual'] = gains['actual'].cumsum()
    gains['cum_non_actual'] = gains['non_actual'].cumsum()
    gains['percent_cum_actual'] = (gains['cum_actual'] / np.max(gains['cum_actual'])).round(2)
    gains['percent_cum_non_actual'] = (gains['cum_non_actual'] / np.max(gains['cum_non_actual'])).round(2)
    gains['if_random'] = np.max(gains['cum_actual']) /10 
    gains['if_random'] = gains['if_random'].cumsum()
    gains['lift'] = (gains['cum_actual'] / gains['if_random']).round(2)
    gains['K_S'] = np.abs( gains['percent_cum_actual'] -  gains['percent_cum_non_actual'] ) * 100
    gains['gain']=(gains['cum_actual']/gains['cum_count']*100).round(2)
    gains = pd.DataFrame(gains)
    return(gains)

createGainsAml(aml1)


def ROC_AUC_AML(my_result,df,target):
    from sklearn.metrics import roc_curve,auc
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    # ROC
    y_actual = df[target].as_data_frame()
    y_pred = my_result.predict(df).as_data_frame()
    fpr = list()
    tpr = list()
    roc_auc = list()
    fpr,tpr,_ = roc_curve(y_actual,y_pred)
    roc_auc = auc(fpr,tpr)
    
    # Precision-Recall
    average_precision = average_precision_score(y_actual,y_pred)

    print('')
    print('   * ROC curve: The ROC curve plots the true positive rate vs. the false rositive sate')
    print('')
    print('	  * The area under the curve (AUC): A value between 0.5 (random) and 1.0 (perfect), measuring the prediction accuracy')
    print('')
    print('   * Recall (R) = The number of true positives / (the number of true positives + the number of false negatives)')
    print('')
    
    # plotting
    plt.figure(figsize=(10,4))

    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr,tpr,color='darkorange',lw=2,label='ROC curve (aare=%0.2f)' % roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=3,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: AUC={0:0.4f}'.format(roc_auc))
    plt.legend(loc='lower right')

    # Precision-Recall
    plt.subplot(1,2,2)
    precision,recall,_ = precision_recall_curve(y_actual,y_pred)
    plt.step(recall,precision,color='b',alpha=0.2,where='post')
    plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title('Precision-Recall curve: PR={0:0.4f}'.format(average_precision))
    plt.show()


ROC_AUC_AML(aml1,test_hex,'loan_default')


# With the understanding that AutoML operates best on "messy" data, so I am uploading the messy data I created last week.

test = pd.read_csv("C:\\Users\\sbortnem\\Desktop\\test.csv")

train = pd.read_csv("C:\\Users\\sbortnem\\Desktop\\train.csv")


variables = pd.DataFrame(train.dtypes).reset_index()
variables.columns = ['name','dtype'] 
variables['var_source'] = variables['name'].str[:2]
variables.var_source.value_counts()


MB_vars = list(variables[variables.var_source =='MB']['name'])
AP_vars = list(variables[(variables.var_source =='AP') & (variables['name']!='AP004') & (variables['name']!='AP001')]['name'])
TD_vars = list(variables[variables.var_source =='TD']['name'])
CR_vars = list(variables[variables.var_source =='CR']['name'])
PA_vars = list(variables[variables.var_source =='PA']['name'])
CD_vars = list(variables[variables.var_source =='CD']['name'])

pred_vars = CR_vars + TD_vars + AP_vars + MB_vars + CD_vars + PA_vars


train_s = train.sample(frac=0.1, random_state=1)
test_s = test.sample(frac=0.1, random_state=1)
train_hex = h2o.H2OFrame(train_s)
test_hex = h2o.H2OFrame(test_s)


aml2 = H2OAutoML(max_runtime_secs = 90, max_models=20, seed=1)
aml2.train(pred_vars,target,training_frame=train_hex)

aml2.leaderboard


pred = aml2.predict(test_hex)
pred.head()


createGainsAml(aml2)

ROC_AUC_AML(aml2,test_hex,'loan_default')


# **Insights:** Overall, AutoML performed best. Because of this, I will use AutoML to perform analysis on the overall database



# ## AutoML on all database
test = pd.read_csv("C:\\Users\\sbortnem\\Desktop\\Fall 2019\\Anomaly Detection - APAN 5420\\Assignment 10\\test.csv")
train = pd.read_csv("C:\\Users\\sbortnem\\Desktop\\Fall 2019\\Anomaly Detection - APAN 5420\\Assignment 10\\train.csv")

variables = pd.DataFrame(train.dtypes).reset_index()
variables.columns = ['name','dtype'] 
variables['var_source'] = variables['name'].str[:2]
variables.var_source.value_counts()

MB_vars = list(variables[variables.var_source =='MB']['name'])
AP_vars = list(variables[(variables.var_source =='AP') & (variables['name']!='AP004') & (variables['name']!='AP001')]['name'])
TD_vars = list(variables[variables.var_source =='TD']['name'])
CR_vars = list(variables[variables.var_source =='CR']['name'])
PA_vars = list(variables[variables.var_source =='PA']['name'])
CD_vars = list(variables[variables.var_source =='CD']['name'])

pred_vars = CR_vars + TD_vars + AP_vars + MB_vars + CD_vars + PA_vars

train_hex_full = h2o.H2OFrame(train)
test_hex_full = h2o.H2OFrame(test)


# Allowing the model to run for thirty minutes.
aml_full = H2OAutoML(max_runtime_secs = 1800, max_models=20, seed=1) 
aml_full.train(pred_vars,target,training_frame=train_hex)

aml_full.leaderboard

pred_full = aml_full.predict(test_hex_full)
pred_full.head()

createGainsAml(aml_full)

ROC_AUC_AML(aml_full,test_hex,'loan_default')


# ## Conclusion 
# Because AutoML performed better on a sample of the data, I selected it to run on all data. The data it ran on was not clean. 
#It resulted in a lift of 2.03 and an AUC of 0.6686.  

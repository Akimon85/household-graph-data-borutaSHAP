#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
from pyathena import connect
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import collections
from collections import Counter
from collections import defaultdict
import os
import datetime
from redshift_connector import connect
import redshift_connector 
import time
import string
from string import punctuation
import boto3
import logger
import logging.config
import io
from logger import logger


# In[144]:


from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from BorutaShap import BorutaShap, load_data
from catboost import CatBoostClassifier


# In[243]:


import shap
from kmodes.kprototypes import KPrototypes
from joblib import Parallel, delayed
from tqdm import tqdm
from kneed import KneeLocator


# ## Athena Query Class

# In[10]:


class QueryAthena:
    def __init__(self, database):
        self.database = database
        self.folder = 'query_results/'
        self.bucket = 'tn-ms-files'
        self.s3_input = 's3://' + self.bucket + '/my_folder_input'
        self.s3_output =  's3://' + self.bucket + '/' + self.folder
        self.region_name = 'us-west-2'
        self.aws_access_key_id = ""
        self.aws_secret_access_key = ""
    def load_conf(self, query):
        try:
            self.client = boto3.client('athena',
                region_name = self.region_name,
                aws_access_key_id = self.aws_access_key_id,
                aws_secret_access_key= self.aws_secret_access_key
            )
            response = self.client.start_query_execution(
                QueryString = query,
                QueryExecutionContext={
                    'Database': self.database
                },
                ResultConfiguration={
                    'OutputLocation': self.s3_output,
                }
            )
            self.filename = response['QueryExecutionId']
            print('Execution ID: ' + response['QueryExecutionId'])
        except Exception as e:
            print(e)
        return response
    def run_query(self, query):
        res = self.load_conf(query)
        try:
            query_status = None
            while query_status == 'QUEUED' or query_status == 'RUNNING' or query_status is None:
                query_status = self.client.get_query_execution(
                        QueryExecutionId=res["QueryExecutionId"]
                )['QueryExecution']['Status']['State']
                logging.info(query_status)
                if query_status == 'FAILED' or query_status == 'CANCELLED':
                    logger.info(self.client.get_query_execution(
                        QueryExecutionId=res["QueryExecutionId"]
                )['QueryExecution']['Status']["StateChangeReason"])
                    raise Exception('Query failed or was cancelled')
    #            time.sleep(10)
            logging.info('Query finished.')
            df = self.obtain_data()
            return df
        except Exception as e:
            print(e)
    def obtain_data(self):
        try:
            self.resource = boto3.resource('s3',
                region_name = self.region_name,
                aws_access_key_id = self.aws_access_key_id,
                aws_secret_access_key= self.aws_secret_access_key
            )
            response = self.resource.Bucket(self.bucket).Object(
                key=self.folder + self.filename + '.csv'
            ).get()
            return pd.read_csv(
                io.BytesIO(response['Body'].read()),
                encoding='utf8'
            )
        except Exception as e:
            print(e)
QA = QueryAthena(database='pixel_ruggable')


# ## Athena Queries

# In[29]:


query1=f"""WITH pix AS (
    SELECT 
        ip_address, 
        COUNT(CASE WHEN event_name = 'order_all' THEN 1 END) AS order_all_count,
        COUNT(CASE WHEN event_name = 'new_order' THEN 1 END) AS new_order_count,
        COUNT(CASE WHEN event_name = 'homepage_view' THEN 1 END) AS homepage_view_count,
        COUNT(CASE WHEN event_name = 'homepage_view_new' THEN 1 END) AS homepage_view_new_count,
        COUNT(CASE WHEN event_name = 'page_view' THEN 1 END) AS page_view_count
    FROM pixel_ruggable.tn_pixel
    WHERE event_date BETWEEN '2024-04-22' AND '2024-06-02' and length(ip_address)>0
    GROUP BY ip_address
    HAVING COUNT(CASE WHEN event_name = 'homepage_view' THEN 1 END) > 0
),
hhid as (
  select * from juice_data.hhid_graph
    where date("$file_modified_time") = (select date(max("$file_modified_time")) from juice_data.hhid_graph)
    and source='TAPAD'
),
match as (
    select a.*, b.uid
    from pix a, hhid b
    where a.ip_address = b.ip
)
select a.*, b.*
from match a join juice_data.experian_tapad_data b
on a.uid = b.uid
"""


# In[30]:


df = QA.run_query(query1) 


# In[32]:


df.shape


# In[33]:


query2=f"""WITH pix AS (
    SELECT DISTINCT ip_address 
    FROM pixel_ruggable.tn_pixel
    WHERE event_date BETWEEN '2024-04-22' AND '2024-06-02' and length(ip_address)>0 AND event_name='homepage_view'
),
hhid as (
  select * from juice_data.hhid_graph
    where date("$file_modified_time") = (select date(max("$file_modified_time")) from juice_data.hhid_graph)
    and source='TAPAD'
),
match as (
    select a.*, b.uid
    from pix a, hhid b
    where a.ip_address = b.ip
)
    SELECT * 
    FROM juice_data.experian_tapad_data
    Where RAND() <0.0015 AND uid NOT IN (Select uid FROM match)
"""
df2 = QA.run_query(query2) 


# ## Redshift Query

# In[275]:


host = 'redshift-dataproc.cgw2jgrknxyx.us-west-2.redshift.amazonaws.com'
port = 5439
database = 'attribution'
user = 'tableau_report_user'
password = 'SKk!!#94nfiLliL3^nD'

conn = redshift_connector.connect(
    host=host,
    port=port,
    database=database,
    user=user,
    password=password
)
cursor = conn.cursor()
query = 'SELECT * FROM audience.experian_data_mapping;'

cursor.execute(query)
data = cursor.fetchall()
colnames = [desc[0] for desc in cursor.description]

cursor.close()
conn.close()

exp_dict = pd.DataFrame(data, columns=colnames)

print(exp_dict)


# ## Data Cleaning

# In[34]:


#Create binary visitor/order target columns & combine datasets

df['visitor']=1
df2['visitor']=0
df = pd.concat([df, df2])
df['order'] = np.where(df['order_all_count']>0,1,0)


# In[122]:


has_second_person = df['person_2_person_id_number'].notnull().astype(int)
df = pd.concat([df, has_second_person.rename('has_second_person')], axis=1)


# In[38]:


ip_counts = df['ip_address'].value_counts()

# Filter IPs with more than 1 occurrence
ips_with_more_than_one_row = ip_counts[ip_counts > 1]

# Get the number of such IPs
num_ips_with_more_than_one_row = len(ips_with_more_than_one_row)

print(num_ips_with_more_than_one_row)


# In[39]:


uid_counts = df2['uid'].value_counts()

# Filter IPs with more than 1 occurrence
uids_more_than_one_row = uid_counts[uid_counts > 1]

# Get the number of such IPs
num_uids_with_more_than_one_row = len(uids_more_than_one_row)

print(num_uids_with_more_than_one_row)


# In[41]:


df.replace('', np.nan, inplace=True)
#df.drop_duplicates(inplace = True)


# In[49]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_cols = df.columns.to_list()
df_cols


# In[53]:


df.drop(columns=[
        'experian_altice_unit_id',
        'address_id','zip_4','post_direction','unit_designator','city_name_abbreviated',
        'state_code','county_code','census_2020_tract_block_group',
        'person_4_person_id_number',
        'person_4_person_type',
        'person_4_combined_age',
        'person_4_gender',
        'person_4_ethnic_group',
        'person_4_ethnic_religion',
        'person_4_ethnic_language_preference',
        'person_4_marital_status',
        'person_4_education_model',
        'person_4_occupation_code',
        'person_5_person_id_number',
        'person_5_person_type',
        'person_5_combined_age',
        'person_5_gender',
        'person_5_ethnic_group',
        'person_5_ethnic_religion',
        'person_5_ethnic_language_preference',
        'person_5_marital_status',
        'person_5_education_model',
        'person_5_occupation_code',
        'person_6_person_id_number',
        'person_6_person_type',
        'person_6_combined_age',
        'person_6_gender',
        'person_6_ethnic_group',
        'person_6_ethnic_religion',
        'person_6_ethnic_language_preference',
        'person_6_marital_status',
        'person_6_education_model',
        'person_6_occupation_code',
        'person_7_person_id_number',
        'person_7_person_type',
        'person_7_combined_age',
        'person_7_gender',
        'person_7_ethnic_group',
        'person_7_ethnic_religion',
        'person_7_ethnic_language_preference',
        'person_7_marital_status',
        'person_7_education_model',
        'person_7_occupation_code',
        'person_8_person_id_number',
        'person_8_person_type',
        'person_8_combined_age',
        'person_8_gender',
        'person_8_ethnic_group',
        'person_8_ethnic_religion',
        'person_8_ethnic_language_preference',
        'person_8_marital_status',
        'person_8_education_model',
        'person_8_occupation_code',
        'person_4_occupation_group_v2',
         'person_5_occupation_group_v2',
         'person_6_occupation_group_v2',
         'person_7_occupation_group_v2',
         'person_8_occupation_group_v2',
         'uid.1', 'match_level_for_geo_data','address_quality_indicator'], inplace=True)


# In[66]:


df.drop(columns=['recipient_reliability_code'], inplace=True)


# In[95]:


df.drop(columns=['person_3_person_id_number',
                 'person_3_person_type',
                 'person_3_combined_age',
                 'person_3_gender',
                 'person_3_ethnic_group',
                 'person_3_ethnic_religion',
                 'person_3_ethnic_language_preference',
                 'person_3_marital_status',
                 'person_3_education_model',
                 'person_3_occupation_code'], inplace=True)


# In[110]:


df.drop(columns = 'person_3_occupation_group_v2', inplace=True)


# In[55]:


'''
def get_mode(ip_subset):
    return ip_subset.mode().iloc[0] if not ip_subset.empty and not ip_subset.mode().empty else None

dd_df = df.groupby('ip').agg(get_mode).reset_index()
'''


# In[12]:


#dd_df.reset_index(drop=True, inplace=True)


# In[111]:


df_cols = df.columns.to_list()
df_cols


# In[129]:


id_cols = ['ip_address',
           'uid',
           'zip_code',
           'city_name',
           'state_abbreviation',
           'person_1_person_id_number', 
           'person_2_person_id_number']

target_cols = ['order_all_count',
             'new_order_count',
             'homepage_view_count',
             'homepage_view_new_count',
             'page_view_count',
              'visitor',
              'order']


# In[115]:


for col in num_cols:
    print(col + ': ')
    print(dd_df[col].isna().sum() / len(dd_df))
    print('-----------------')


# In[92]:


for col in num_cols:
    if col not in id_cols:
        print(col +': ' + str(dd_df[col].unique()))


# In[81]:


for col in ['person_1_combined_age','person_2_combined_age']:
    dd_df[col] = dd_df[col].str.extract('(\d+)', expand=False)
    dd_df[col] = pd.to_numeric(dd_df[col])


# In[83]:


def find_bool_cols(column):
    allowed_values = {'U','Y'}
    unique_values = set(column.dropna().unique())
    return unique_values.issubset(allowed_values)

bool_cols = [col for col in dd_df.columns if find_bool_cols(dd_df[col])]


# In[85]:


dd_df[bool_cols] = dd_df[bool_cols].applymap(lambda x: 1 if x == 'Y' else 0)


# In[187]:


num_cols = ['num_of_persons_in_living_unit',
            'num_of_adults_in_living_unit',
            'num_of_child_in_living_unit',
            'mor_bank_doityourselfers',
            'length_of_residence',
            'buyer_freq_online_shopper',
            'food_freq_family_rest_diner',
            'food_freq_fast_food_diner',
            'food_general_family_rest',
            'food_occasional_family_rest',
            'food_occasional_fast_food_rest',
            'invest_investor_high_value',
            'invest_investor_midvalue',
            'lifestyle_interest_in_religion',
            'person_1_combined_age',
            'person_2_combined_age',
             #'has_second_person'
           ]


# In[133]:


cat_cols = set(df_cols) - set().union(*[num_cols, bool_cols, id_cols, target_cols])


# In[ ]:


cat_cols = list(cat_cols)


# In[190]:


categorical_columns = set().union(*[cat_cols, bool_cols])
categorical_columns = list(categorical_columns)


# In[155]:


dd_df[cat_cols] = dd_df[cat_cols].astype('str')
dd_df[cat_cols] = dd_df[cat_cols].fillna('NaN')


# In[176]:


X = dd_df.drop(columns=id_cols + target_cols)
for col in cat_cols:
    X[col] = X[col].astype('str')
y_visit = dd_df['visitor']
y_order = dd_df['order']

# Split the data into training and testing sets
X_train, X_test, y_visit_train, y_visit_test = train_test_split(X, y_visit, test_size=0.2, random_state=66)
X_train, X_test, y_order_train, y_order_test = train_test_split(X, y_order, test_size=0.2, random_state=66)


# In[157]:


# Initialize CatBoost classifier
catboost_clf = CatBoostClassifier(silent=False)

# Run Boruta with CatBoost with order status as target
boruta_selector_order = BorutaShap(model=catboost_clf, importance_measure='shap', classification=True)
boruta_selector_order.fit(X=X_train, y=y_order_train, n_trials=100, random_state=66, normalize=True, verbose=True)

boruta_selector_order.plot(which_features='all')


# In[160]:


boruta_selector_order.plot(which_features='all',figsize=(16,8),
            y_scale='log')


# In[162]:


# Run BorutaShap with CatBoost with visit status as target
boruta_selector_visit = BorutaShap(model=catboost_clf, importance_measure='shap', classification=True)
boruta_selector_visit.fit(X=X_train, y=y_visit_train, n_trials=100, random_state=66, normalize=True, verbose=True)

boruta_selector_visit.plot(which_features='all', X_size=12, figsize=(12,8),
            y_scale='log')


# In[166]:


visitor_feature_importance = catboost_clf.get_feature_importance()


# In[167]:


visitor_feature_importance


# In[165]:


boruta_selector_visit.plot(which_features='all', X_size=12, figsize=(24,8),
            y_scale='log')


# In[198]:


visitor_features = [
    'person_2_combined_age', 
    'dwelling_type', 
    'person_1_ethnic_language_preference', 
    'person_1_ethnic_group', 
    'behavbnk_intrst_in_gourmet_cooking', 
    'food_occasional_fast_food_rest', 
    'num_of_persons_in_living_unit', 
    'person_2_education_model', 
    'num_of_adults_in_living_unit', 
    'behavbnk_sweepstakes_gambling', 
    'food_freq_family_rest_diner', 
    'estimate_hh_income_range_code_v6', 
    'estimated_current_home_value_range', 
    'person_1_occupation_group_v2', 
    'buyer_freq_online_shopper', 
    'invest_investor_high_value', 
    'combined_homeowner_renter', 
    'person_1_combined_age', 
    'household_composition', 
    'child_age_0_3', 
    'length_of_residence', 
    'child_age_7_9', 
    'person_1_education_model', 
    'behavbnk_intrst_in_sports', 
    'behavbnk_intrst_in_reading',
    'person_2_occupation_group_v2', 
    'person_2_gender', 
    'food_general_family_rest', 
    'food_occasional_family_rest', 
    'lifestyle_interest_in_religion', 
    'num_of_child_in_living_unit', 
    'person_1_marital_status', 
    'srvy_acty_magazine_food_wine_cooking', 
    'srvy_lfstyl_pets_own'
]
visitor_num_feats = list(set(visitor_features) & set(num_cols))
visitor_cat_cols = list(set(visitor_features) & set(categorical_columns))


# In[204]:


for col in X_train_std.columns:
    print(col)
    print(X_train_std[col].isna().sum())


# In[213]:


X_train_std = X_train[visitor_features].drop(columns='person_2_combined_age').copy()
# Standardize numerical columns
visitor_num_feats.remove('person_2_combined_age')
scaler = StandardScaler()
X_train_std[visitor_num_feats] = scaler.fit_transform(X_train_std[visitor_num_feats])
cat_indices = [X_train_std.columns.get_loc(col) for col in visitor_cat_cols]


# In[219]:


concat_1 = pd.concat([X_train_std, y_visit_train.rename('y1')], axis=1)


# In[220]:


train_subset_df = pd.concat([concat_1, y_order_train.rename('y2')], axis=1)


# In[223]:


left_out_df, sample_df = train_test_split(train_subset_df, test_size=0.1, random_state=66)
sample_x = sample_df[X_train_std.columns]
sample_y_visit = sample_df['y1']
sample_y_order = sample_df['y2']


# In[229]:


K = range(1, 10)
cost = []

# Fit K-Prototypes
def fit_kprototypes(k, X, cat_indices):
    kproto = KPrototypes(n_clusters=k, random_state=66, init='Cao')
    kproto.fit(X, categorical=cat_indices)
    return k, kproto.cost_

with tqdm(total=len(K)) as pbar:
    # Parallelize the computation across all available CPU cores
    results = Parallel(n_jobs=-1)(delayed(fit_kprototypes)(k, sample_x, cat_indices) for k in K)
    # Update the progress bar after each parallel job completes
    pbar.update(len(K))

costs = [cost for k, cost in results]
    
plt.plot(K, costs, 'bx-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal K (Parallel)')
plt.show()


# In[248]:


cost_df = pd.DataFrame(costs, columns=['cost'])
cost_df = cost_df.reset_index()
cost_df = cost_df.rename(columns={'index':'k','cost':'cost'})


# In[253]:


cost_knee = KneeLocator(x=cost_df['k'], y=cost_df['cost'], S=0.1, curve='convex', direction='decreasing', online=True)
K_cost = cost_knee.elbow
print('elbow at k =', f'{K_cost:.0f} clusters')


# In[256]:


kproto3 = KPrototypes(n_clusters=3, random_state=66, init='Huang', verbose=True, n_jobs=10)
c3 = kproto3.fit_predict(sample_x, categorical=cat_indices)


# In[258]:


np.unique(c3, return_counts=True)


# In[260]:


kproto4 = KPrototypes(n_clusters=4, random_state=66, init='Huang', verbose=True, n_jobs=10)
c4 = kproto4.fit_predict(sample_x, categorical=cat_indices)


# In[261]:


np.unique(c4, return_counts=True)


# In[263]:


train_subset_df.shape


# In[265]:


sample_x['cluster_3']=c3


# In[266]:


sample_x['cluster_4']=c4


# In[270]:


sample_x_original = sample_x.copy()
sample_x_original[visitor_num_feats] = scaler.inverse_transform(sample_x_original[visitor_num_feats])


# In[271]:


sample_x_original['cluster_3']=c3
sample_x_original['cluster_4']=c4
sample_x_original['visit'] = sample_y_visit
sample_x_original['order'] = sample_y_order


# In[276]:


exp_dict


# In[289]:


common_columns = set(exp_dict['experian_field'].unique()).intersection(sample_x_original.columns)
sample_x_mapped = sample_x_original.copy()
for column in common_columns:
    mapping_dict = exp_dict[exp_dict['experian_field'] == column].set_index('experian_value')['experian_mapped_value'].to_dict()
    sample_x_mapped[column] = sample_x_original[column].map(mapping_dict).fillna(sample_x_original[column])


# In[290]:


agg_dict = {col: 'mean' for col in visitor_num_feats}
agg_dict.update({col: lambda x: x.mode()[0] for col in visitor_cat_cols})

cluster3_results = sample_x_mapped.groupby('cluster_3').agg(agg_dict).reset_index()


# In[291]:


cluster3_results.T


# In[295]:


sample_x_mapped.groupby('cluster_3').agg({'order':'mean','visit':'mean'}).reset_index()


# In[296]:


sample_x_mapped.groupby('cluster_4').agg({'order':'mean','visit':'mean'}).reset_index()


# In[292]:


agg_dict = {col: 'mean' for col in visitor_num_feats}
agg_dict.update({col: lambda x: x.mode()[0] for col in visitor_cat_cols})

cluster4_results = sample_x_mapped.groupby('cluster_4').agg(agg_dict).reset_index()


# In[307]:


cluster4_results.T.sort_index()


# In[299]:


common_columns = set(exp_dict['experian_field'].unique()).intersection(dd_df.columns)
dd_df_mapped = dd_df.copy()
for column in common_columns:
    mapping_dict = exp_dict[exp_dict['experian_field'] == column].set_index('experian_value')['experian_mapped_value'].to_dict()
    dd_df_mapped[column] = dd_df[column].map(mapping_dict).fillna(dd_df[column])


# In[301]:


dd_df_mapped.to_csv('rug_exp_mapped.csv')


# In[302]:


sample_x_mapped.to_csv('rug_exp_clusters.csv')


# In[141]:


# Initialize CatBoost classifier
catboost_clf = CatBoostClassifier(silent=False, cat_features=cat_cols)

# Run Boruta with CatBoost with order status as target
boruta_selector_order = BorutaPy(estimator=catboost_clf, n_estimators='auto', verbose=2, random_state=66, perc=85)
boruta_selector_order.fit(X_train.values, y_order_train.values)

feature_names = X.columns
feature_importances_catboost = catboost_clf.get_feature_importance()
feature_ranks = list(zip(feature_names, feature_importances_catboost, boruta_selector_order.ranking_, boruta_selector_order.support_, boruta_selector_order.support_weak_ )) 
boruta_results = pd.DataFrame(feature_ranks, columns=['feature_name', 'cb_feature_importance', 'rank', 'confirmed','tentative'])
print(boruta_results)
boruta_results.to_csv('rug_exp_order_features.csv')


# In[ ]:


# Run Boruta with CatBoost with visit status as target
boruta_selector_visit = BorutaPy(estimator=catboost_clf, n_estimators='auto', verbose=2, random_state=66, perc=85)
boruta_selector_visit.fit(X_train.values, y_visit_train.values)

feature_names = X.columns
feature_importances_catboost = catboost_clf.get_feature_importance()
feature_ranks = list(zip(feature_names, feature_importances_catboost, boruta_selector_visit.ranking_, boruta_selector_visit.support_, boruta_selector_visit.support_weak_ )) 
boruta_results = pd.DataFrame(feature_ranks, columns=['feature_name', 'cb_feature_importance', 'rank', 'confirmed','tentative'])
print(boruta_results)
boruta_results.to_csv('rug_exp_visit_features.csv')


# In[88]:


sns.histplot(data=df[df['order_all_count']>0], x='order_all_count', hue='order')


# In[ ]:





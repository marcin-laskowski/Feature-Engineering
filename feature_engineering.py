'''
INTRODUCTION TO AUTOMATED FEATURE ENGINEERING

In this python script, it will be implemented Featuretools, an open-source Python library for
automatically creating features with relational data (where the data is in structured tables).
Although there are now many efforts working to enable automated model selection and hyperparameter
tuning, there has been a lack of automating work on the feature engineering aspect of the pipeline.
This library seeks to close that gap and the general methodology has been proven effective in both
machine learning competitions with the data science machine and business use cases.

This is a powerful method which allows us to overcome the human limits of time and imagination to
create many new features from multiple tables of data. Featuretools is built on the idea of deep
feature synthesis, which means stacking multiple simple feature primitives - aggregations and
transformations - to create new features. Feature engineering allows us to combine informatio
across many tables into a single dataframe that we can then use for machine learning model training.
Finally, the next step after creating all of these features is figuring out which ones are
important.
'''

# Run this if featuretools is not already installed
# !pip install -U featuretools


# =============================== LIBRARIES =======================================================
# pandas and numpy for data manipulation
import pandas as pd
import numpy as np
# featuretools for automated feature engineering
import featuretools as ft
# ignore warnings from pandas
import warnings
warnings.filterwarnings('ignore')


# =============================== READ DATA ========================================================

# Read in the data
# 'parse_dates' recogenize the date
clients = pd.read_csv('data/clients.csv', parse_dates=['joined'])
loans = pd.read_csv('data/loans.csv', parse_dates=['loan_start', 'loan_end'])
payments = pd.read_csv('data/payments.csv', parse_dates=['payment_date'])


# =============================== READ DATA ========================================================

# method 'head' show to 5 positions of the dataset
print(clients.head())
# method 'sample' show ranodm n raws of the dataset
loans.sample(10)
payments.sample(10)


# =============================== MANUAL FEATURE ENIGNEERING =======================================

# Create a month column
clients['join_month'] = clients['joined'].dt.month

# Create a log of income column
clients['log_income'] = np.log(clients['income'])

print(clients.head())

# To incorporate information about the other tables, we use the df.groupby method, followed by
# a suitable aggregation function, followed by df.merge.
# For example, let's calculate the average, minimum, and maximum amount of previous loans for each
# client. In the terms of featuretools, this would be considered an aggregation feature primitive
# because we using multiple tables in a one-to-many relationship to calculate aggregation figures.

# Groupby client id and calculate mean, max, min previous loan size
stats = loans.groupby('client_id')['loan_amount'].agg(['mean', 'max', 'min'])
stats.columns = ['mean_loan_amount', 'max_loan_amount', 'min_loan_amount']
print(stats.head())

# Merge with clients dataframe
clients.merge(stats, left_on='client_id', right_index=True, how='left').head(10)
print(clients.head(10))


# =================================================================================================
# =============================== FEATURETOOLS ====================================================
# =================================================================================================

# The first part of Featuretools to understand is an entity. This is simply a table, or in pandas,
# a DataFrame. We corral multiple entities into a single object called an EntitySet. This is just
# a large data structure composed of many individual entities and the relationships between them.

# ==================================== ENTITES ====================================================

# An entity is simply a table, which is represented in Pandas as a dataframe. Each entity must
# have a uniquely identifying column, known as an index. For the clients dataframe, this is the
# client_id because each id only appears once in the clients data. In the loans dataframe,
# client_id is not an index because each id might appear more than once. The index for this
# dataframe is instead loan_id.

# When we create an entity in featuretools, we have to identify which column of the dataframe is
# the index. If the data does not have a unique index we can tell featuretools to make an index
# for the entity by passing in make_index = True and specifying a name for the index. If the data
# also has a uniquely identifying time index, we can pass that in as the time_index parameter.

# Featuretools will automatically infer the variable types (numeric, categorical, datetime) of
# the columns in our data, but we can also pass in specific datatypes to override this behavior.
# As an example, even though the repaid column in the loans dataframe is represented as an integer,
# we can tell featuretools that this is a categorical feature since it can only take on two
# discrete values. This is done using an integer with the variables as keys and the feature types
# as values.

# Create new EntitySet
es = ft.EntitySet(id='clients')

# Create an entity from the client DataFrame
# This dataframe already has an index and a time index
es = es.entity_from_dataframe(entity_id='clients', dataframe=clients, index='client_id',
                              time_index='joined')
# Create an entity from the loans DataFrame
# This DataFrame already has an index and a time index
es = es.entity_from_dataframe(entity_id='loans', dataframe=loans,
                              variable_types={'repaid': ft.variable_types.Categorical},
                              index='loan_id', time_index='loan_start')
# Create an entity from the payments DataFrame
# This does not yet have a unique index
es = es.entity_from_dataframe(entity_id='payments', dataframe=payments,
                              variable_types={'missed': ft.variable_types.Categorical},
                              make_index=True, index='payment_id', time_index='payment_date')

# Summary of the entity with the data.
# All three entities have been successfully added to the EntitySet
print(es)

# We can access any of the entities using Python dictionary syntax.
print(es['loans'])
print(es['payments'])


# ==================================== RELATIONSHIPS ===============================================

# The most intuitive way to think of relationships is with the parent to child analogy:
# a parent-to-child relationship is one-to-many because for each parent, there can be multiple
# children. The 'client' dataframe is therefore the parent of the 'loans' dataframe because while
# there is only one row for each client in the client dataframe, each client may have several
# previous loans covering multiple rows in the loans dataframe. Likewise, the 'loans' dataframe is
# the parent of the 'payments' dataframe because each loan will have multiple payments.

# These relationships are what allow us to group together datapoints using aggregation primitives
# and then create new features. As an example, we can group all of the previous loans associated
# with one client and find the average loan amount.

# To define relationships, we need to specify the parent variable and the child variable.
# This is the variable that links two entities together. In our example, the client and loans
# dataframes are linked together by the client_id column. Again, this is a parent to child
# relationship because for each client_id in the parent client dataframe, there may be multiple
# entries of the same client_id in the child loans dataframe.


# We codify relationships in the language of featuretools by specifying the parent variable and
# then the child variable. After creating a relationship, we add it to the EntitySet.

# Relationship between clients and previous loans
r_client_previous = ft.Relationship(es['clients']['client_id'], es['loans']['client_id'])
# Add the realtionship to the entity set
es = es.add_relationship(r_client_previous)


# The relationship has now been stored in the entity set. The second relationship is between
# the loans and payments. These two entities are related by the loan_id variable.

# Relationship between previous loans and previous payments
r_payments = ft.Relationship(es['loans']['loan_id'], es['payments']['loan_id'])
# Add relationship to the entity set
es = es.add_relationship(r_payments)

print(es)


# ==================================== FUTURE PRIMITIVES ===========================================

# A feature primitive is an operation applied to data to create feature. These represent very simple
# calculations that can be stacked on top of each other to create complex features.
# Feature primitives fail into two categories:
# 1. Aggregation: function that groups together child datapoints for each parent and then calculates
# a statistic such as mean, min, max or standard deviation. An example is calcualting the maximum
# loan amount for each client. An aggregation works across multiple tables using relationships
# between tables.
# 2. Transformation: an operation applied to one or more columns in a single table. An example
# would be extracting the day from dates, or finiding the difference between two columns in one
# table.

# present list of primitives
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(10)
primitives[primitives['type'] == 'transform'].head(10)


# In featuretools it is also possible to create your own primitives.
# the 'ft.dfs' function (which stands for: deep feature synthesis). In this function, we specify the
# entityset to use; the target_entity, which is the dataframe we want to make the features for. The
# agg_primitives which are the aggregation feature primitives; and the trans_primitives which are
# transformation primitives to apply.

# Create new features using specified primitives
features, feature_names = ft.dfs(entityset=es, target_entity='clients',
                                 agg_primitives=['mean', 'max', 'percent_true', 'last'],
                                 trans_primitives=['years', 'month', 'subtract', 'divide'])

pd.DataFrame(features['MONTH(joined)'].head())
pd.DataFrame(features['MEAN(payments.payment_amount)'].head())
features.head()


# ================================= DEEP FEATURE SYNTHESIS =========================================

# While feature primitives are useful by themselves, the main benefit of using featuretools arises
# when we stack primitives to get deep features. The depth of a feature is simply the number of
# primitives required to make a feature. So, a feature that relies on a single aggregation would be
# a deep feature with a depth of 1, a feature that stacks two primitives would have a depth of 2
# and so on. The idea itself is lot simpler than the name "deep feature synthesis" implies.
# (more in the link: https://docs.featuretools.com/automated_feature_engineering/afe.html)


# the MEAN(loans.loan_amount) feature has a depth of 1 because it is made by applying a single
# aggregation primitive. This feature represents the average size of a client's previous loans.

# show a feature with a depth of 1
pd.DataFrame(features['MEAN(loans.loan_amount)'].head(10))

# As well scroll through the features, we see a number of features with a depth of 2. For example,
# the LAST(loans.(MEAN(payments.payment_amount))) has depth = 2 because it is made by stacking two
# feature primitives, first an aggregation and then a transformation. This feature represents
# the average payment amount for the last (most recent) loan for each client.

# Show a feature with depth of 2
pd.DataFrame(features['LAST(loans.MEAN(payments.payment_amount))'].head(10))

# We can create features of arbitrary depth by stacking more primitives. However, it is important to
# consider the fact that primitives with depth more than 2 become very convoluted to understand.


# ============================== AUTOMATED DEEP FEATURE SYNTHESIS ==================================

# In addition to manually specifying aggregation and transformation feature primitives, we can let
# featuretools automatically generate many new features. We do this by making the same ft.dfs
# function call, but without passing in any primitives. We just set the max_depth parameter and
# featuretools will automatically try many all combinations of feature primitives to the ordered
# depth

# perform deep feature synthesis without specifying primitives
features, feature_name = ft.dfs(entityset=es, target_entity='clients', max_depth=2)
features.iloc[:, 4:].head()

# Deep feature synthesis has created 90 new features out of the existing data! While we could have
# created all of these manually, I am glad to not have to write all that code by hand. The primary
# benefit of featuretools is that it creates features without any subjective human biases. Even a
# human with considerable domain knowledge will be limited by their imagination when making new
# features (not to mention time). Automated feature engineering is not limited by these factors
# (instead it's limited by computation time) and provides a good starting point for feature
# creation. This process likely will not remove the human contribution to feature engineering
# completely because a human can still use domain knowledge and machine learning expertise to
# select the most important features or build new features from those suggested by automated deep
# feature synthesis.

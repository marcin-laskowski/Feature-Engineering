'''
INTRODUCTION TO AUTOMATED FEATURE ENGINEERING

In this python script, it will be implemented Featuretools, an open-source Python library for
automatically creating features with relational data (where the data is in structured tables).
Although there are now many efforts working to enable automated model selection and hyperparameter
tuning, there has been a lack of automating work on the feature engineering aspect of the pipeline.
This library seeks to close that gap and the general methodology has been proven effective in both
machine learning competitions with the data science machine and business use cases.
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

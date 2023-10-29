"""
Apriori with the mlxtend library
"""

import numpy as np
from pandas import DataFrame
from mlxtend.frequent_patterns import apriori, association_rules


# make data / df
m,n = 10, 5
p = 0.25
X = np.random.choice([0,1], size=(m,n), replace=True, p=(1.0-p, p)).astype(np.bool_)
df = DataFrame(X, columns=[chr(97+j) for j in range(n)], index=[f"t{i+1}" for i in range(m)])
df.index.name = "transaction"
df.columns.name = "items"


df_itemsets = apriori(df, min_support=0.2, use_colnames=True)
df_rules = association_rules(df_itemsets, metric='confidence', min_threshold=0.5)

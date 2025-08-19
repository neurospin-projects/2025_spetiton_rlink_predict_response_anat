import numpy as np
import pandas as pd


response_filename = "/neurospin/rlink/PUBLICATION/rlink-ecrf/dataset-outcome_version-4.tsv"
participants_filename = "/neurospin/rlink/participants.tsv"

response_df = pd.read_csv(response_filename, sep='\t')
participants_df = pd.read_csv(participants_filename, sep='\t')

assert participants_df.shape[0] == 168
np.sort(participants_df["ses-M00_center"].unique())

# array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
#       14., 15., 16., nan])

assert response_df.shape[0] == 138
response_df.columns
resp_defs =  ['Status.per.protocol', 'Status.ITT', 'Response.Status.at.end.of.follow.up']
# ['participant_id', 'FollowUp_Duration (duration of Li treatment)',
#       'Response.Status.at.end.of.follow.up', 'Status.per.protocol',
#       'Status.ITT'],

response_df["Response.Status.at.end.of.follow.up"].value_counts()
"""
Response.Status.at.end.of.follow.up
PaR    56
GR     49
NR     33
UC     21
Name: count, dtype: int64
"""

for resp_def in resp_defs:
    print(response_df[resp_def].value_counts())

count = pd.concat([response_df[resp_def].value_counts() for resp_def in resp_defs], axis=1)
count.columns = resp_defs
print(count.T)
"""
                                      UC  PaR  GR  NR
Status.per.protocol                  100   30  25   4
Status.ITT                            58   49  40  12
Response.Status.at.end.of.follow.up   21   56  49  33
"""
count.sum(axis=0) == 159

import re
import pandas as pd

UNIQNAME = 'TENGDANN'

raw_table = pd.read_csv('dates.txt', sep = '\t', header = None)
raw_table.rename({0: 'Row', 1: 'Text'}, axis = 'columns', inplace = True)
raw_table.set_index(['Row'], inplace = True)

# REQUIRES: a row from pandas dataframe that contains a valid text.
# MODIFIES: nothing
# EFFECTS: returns a normalized date as per the Assignment 1 specification
def date_normalizer(row):
    # Assume all dates in xx/xx/xx format are mm/dd/yy.
    # Assume dates in xx/xx format are mm/yy.
    # Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
    # If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
    # If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).

    pat1 = re.compile('\d{,2}')
    pat2 = 
    pat3 = 
    pat4 = 
    pass

raw_table['Date'] = raw_table.apply(date_normalizer, axis = 1)
print(raw_table.head(5))
import re
import pandas as pd

UNIQNAME = 'TENGDANN'
filename = 'LHS712-Assg1-%s.txt' % UNIQNAME
MONTHS = '(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
MONTH_DICT = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

raw_table = pd.read_csv('dates.txt', sep = '\t', header = None)
raw_table.rename({0: 'Row', 1: 'Text'}, axis = 'columns', inplace = True)
raw_table.set_index(['Row'], inplace = True)

# REQUIRES: a row from pandas dataframe that contains a valid text.
# MODIFIES: nothing
# EFFECTS: parses text for dates; returns list of normalized dates found in row
def date_parser(row):
    # Assume all dates in xx/xx/xx format are mm/dd/yy.
    # Assume dates in xx/xx format are mm/yy.
    # Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
    # If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
    # If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).

    pat1 = r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
    pat2 = r'(\d{1,2})[/-](\d{2,4})'
    pat3 = r'(\d{1,2})\s?(%s[a-z]*)\s?(?:\d{1,2})?,? (\d{2,4})' % MONTHS
    pat4 = r'(\d{4})'

    if len(re.findall(pat1, row['Text'])) != 0:
        raw_pat = re.findall(pat1, row['Text'])
        month = raw_pat[0][0]
        day = raw_pat[0][1]
        year = raw_pat[0][2]
        
        return date_normalizer(year, month, day)
    elif len(re.findall(pat2, row['Text'])) != 0:
        raw_pat = re.findall(pat2, row['Text'])
        month = raw_pat[0][0]
        year = raw_pat[0][1]

        return date_normalizer(year, month)
    elif len(re.findall(pat3, row['Text'])) != 0:
        raw_pat = re.findall(pat3, row['Text'])
        day = raw_pat[0][0]
        month = raw_pat[0][1]
        year = raw_pat[0][2]
        
        return date_normalizer(year, month, day)
    elif len(re.findall(pat4, row['Text'])) != 0:
        raw_pat = re.findall(pat4, row['Text'])
        year = raw_pat[0]

        return date_normalizer(year)
    else:
        return 'No date found?'

# REQUIRES: valid string for year
# MODIFIES: nothing
# EFFECTS: returns a normalized date as per the Assignment 1 spec
def date_normalizer(year, month = '01', day = '01'):
    # Normalize year
    if len(year) != 4:
        year = '19' + year

    # Normalize month
    if len(month) != 2 and month.isdigit():
        month = '0' + month
    elif month.isdigit() is False:
        month = MONTH_DICT[month[0:3]]

    # Normalize day
    if len(day) != 2:
        day = '0' + day

    return '%s-%s-%s' % (year, month, day)

raw_table['Date'] = raw_table.apply(date_parser, axis = 1)
raw_table.to_csv(filename, sep = '\t', columns = ['Date'])
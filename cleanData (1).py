# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:19:35 2021
"""

import pandas as pd
import re

# returnType: i return integer, f return float
# removeStr: pass regex pattern to remove
def cleanData(var, returnType="", removeStr=""):
    
    if not isinstance(var, str):
        var = str(var)
        
    var.lstrip().lstrip()
    
    if removeStr != "":
        var = re.sub(removeStr, "", var)
    
    if returnType == "i":
        if len(var) > 0:
            var = int(var)
        else:
            var = 0
    elif returnType == "f":
        if len(var) > 0:
            var = float(var)
        else:
            var = 0.0
    
    return var


# A=1, B=2...
def convertGrade(x):
    return ord(x)-64


# Jan-90 -> 1990;  Feb-01 -> 2001
def earliestCrLine(x):
    intX = int(x[len(x)-2:])
    if intX < 50:
        intX += 2000
    else:
        intX += 1900
        
    return intX


def dtiBin(x):
    if x <= 200:
        return "10"
    elif x <= 400:
        return "30"
    elif x <= 600:
        return "50"
    elif x <= 800:
        return "70"
    else:
        return "90"
        

def loanStatus(x):
    if x == "Default" or x == "Charged Off":
        return "Dafault"
    elif x == "Current" or x == "Fully Paid":
        return "Paid"
    else:
        return "Late"


def homeOwnershipOther(x):
    if x == "ANY" or x == "NONE":
        return "OTHER"
    else:
        return x
    

def dfFillna(df, aList):
    for item in aList:
        df[item] = df[item].fillna(0)
    
    
columnAll = ['funded_amnt', 'term', 'int_rate', 'installment', 'grade',
             'emp_length', 'home_ownership', 'annual_inc', 'loan_status', 'purpose',
             'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low',
             'fico_range_high', 'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_util',
             'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_prncp', 'total_rec_int',
             'total_rec_late_fee', 'application_type', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq',
             'tot_coll_amt', 'tot_cur_bal', 'open_act_il', 'il_util', 'inq_fi',
             'inq_fi', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mort_acc',
             'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl',
             'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
             'num_tl_30dpd', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens']

columnDPI = ['funded_amnt', 'term', 'int_rate', 'installment', 'grade',
                 'emp_length', 'home_ownership', 'annual_inc', 'loan_status', 'dti',
                 'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'mths_since_last_delinq', 'total_rec_late_fee',
                 'application_type', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq', 'il_util',
                 'bc_util', 'chargeoff_within_12_mths', 'num_accts_ever_120_pd', 'num_sats', 'num_tl_120dpd_2m',
                 'num_tl_30dpd', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens']

#columnTest = ['num_tl_30dpd', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens']

dfAll = pd.read_csv("accepted_2007_to_2018Q4.csv", usecols = columnAll)
#dfAll = pd.read_csv("Sample Data.csv", usecols = columnTest)

#print(dfAll.shape[0])
###   Clean Up   ###
dfAll = dfAll[dfAll['funded_amnt'].notna()]
dfAll = dfAll[dfAll['earliest_cr_line'].notna()]
dfAll = dfAll[dfAll['annual_inc']>0]
dfAll['term']  = dfAll['term'].apply(lambda x: cleanData(x, "i", "\D"))
dfAll['grade'] = dfAll['grade'].apply(lambda x: convertGrade(x))
dfAll['emp_length'] = dfAll['emp_length'].apply(lambda x: cleanData(x, "i", "\D"))
dfAll['earliest_cr_line'] = dfAll['earliest_cr_line'].apply(lambda x: earliestCrLine(x))
dfAll['total_pymnt'] = dfAll['total_pymnt'].apply(lambda x: float("{:.2f}".format(x)))

fillNAList = ["mths_since_last_delinq", "annual_inc_joint", "dti_joint", "open_act_il", "il_util",
             "inq_fi", "bc_util", "num_tl_120dpd_2m", "num_rev_accts", "revol_util"]
dfFillna(dfAll, fillNAList)

###   Combine columns and data   ###
dfAll['home_ownership'] = dfAll['home_ownership'].apply(lambda x: homeOwnershipOther(x))
dfAll['loan_status'] = dfAll['loan_status'].apply(lambda x: loanStatus(x))

dfPartial = dfAll[columnDPI].copy()
dfPartial['loan_status_dpi'] = dfPartial['dti'].apply(lambda x: dtiBin(x)) + "_" + dfPartial['loan_status']

dfBin = dfAll[columnDPI].copy()
bin_labels_5 = ['10_', '30_', '50_', '70_', '90_']
dfBin['loan_status_dpi'] = pd.qcut(dfBin['dti'],
                              q=[0, .2, .4, .6, .8, 1],
                              labels=bin_labels_5)
dfBin['loan_status_dpi'] = dfBin['loan_status_dpi'].astype(str) + dfBin['loan_status']

#print(dfPartial)
#print(dfBin.dtypes)

dfAll.to_csv('columnAll.csv')
dfPartial.to_csv('columnDPI.csv')
dfBin.to_csv('columnBin.csv')
###############################################
##  @author: leonlwang@tencent.com
###############################################
#install.packages("DescTools") for KappaM
library(readxl) 
library(tidyverse) # to read sub-part of excel
library(DescTools) # to use KappaM

setwd("D:/git/PDL1_SP142_Breast_IC_score")
sink("./result_Calc/Calc_FKS_CI95_in_RStudio.txt")   # start to save log

###############################################
#################### Class 4 ##################
###############################################
## [1] Class 4, exp 1
path_array_raw="./IC_Score_Pathologist/ic_table_4class_exp1.xlsx"
data_raw = read_excel(path_array_raw,sheet = 1, col_names = TRUE, col_types = NULL, na = "", skip = 0)

data_all  = data_raw %>% select(2:32)
data_high = data_raw %>% select(2:12)
data_mid  = data_raw %>% select(13:22)
data_low  = data_raw %>% select(23:32)

print(path_array_raw)
print("data_all")
KappaM(data_all,  conf.level=0.95)   # Fleiss' Kappa and confidence intervals
print("data_high")
KappaM(data_high, conf.level=0.95)   
print("data_mid")
KappaM(data_mid,  conf.level=0.95)   
print("data_low")
KappaM(data_low,  conf.level=0.95)



## [2] Class 4, exp 2
path_array_raw="./IC_Score_Pathologist/ic_table_4class_exp2.xlsx"
data_raw = read_excel(path_array_raw,sheet = 1, col_names = TRUE, col_types = NULL, na = "", skip = 0)

data_all  = data_raw %>% select(2:32)
data_high = data_raw %>% select(2:12)
data_mid  = data_raw %>% select(13:22)
data_low  = data_raw %>% select(23:32)

print(path_array_raw)
print("data_all")
print("data_high")
print("data_mid")
print("data_low")
KappaM(data_all,  conf.level=0.95)   # Fleiss' Kappa and confidence intervals
KappaM(data_high, conf.level=0.95)   
KappaM(data_mid,  conf.level=0.95)   
KappaM(data_low,  conf.level=0.95)



## [3] Class 4, exp 3
path_array_raw="./IC_Score_Pathologist/ic_table_4class_exp3.xlsx"
data_raw = read_excel(path_array_raw,sheet = 1, col_names = TRUE, col_types = NULL, na = "", skip = 0)

data_all  = data_raw %>% select(2:32)
data_high = data_raw %>% select(2:12)
data_mid  = data_raw %>% select(13:22)
data_low  = data_raw %>% select(23:32)

print(path_array_raw)
print("data_all")
print("data_high")
print("data_mid")
print("data_low")
KappaM(data_all,  conf.level=0.95)   # Fleiss' Kappa and confidence intervals
KappaM(data_high, conf.level=0.95)   
KappaM(data_mid,  conf.level=0.95)   
KappaM(data_low,  conf.level=0.95)


###############################################
#################### Class 2 ##################
###############################################
## [1] Class 2, exp 1
path_array_raw="./IC_Score_Pathologist/ic_table_2class_exp1.xlsx"
data_raw = read_excel(path_array_raw,sheet = 1, col_names = TRUE, col_types = NULL, na = "", skip = 0)

data_all  = data_raw %>% select(2:32)
data_high = data_raw %>% select(2:12)
data_mid  = data_raw %>% select(13:22)
data_low  = data_raw %>% select(23:32)

print(path_array_raw)
print("data_all")
print("data_high")
print("data_mid")
print("data_low")
KappaM(data_all,  conf.level=0.95)   # Fleiss' Kappa and confidence intervals
KappaM(data_high, conf.level=0.95)   
KappaM(data_mid,  conf.level=0.95)   
KappaM(data_low,  conf.level=0.95)



## [2] Class 2, exp 2
path_array_raw="./IC_Score_Pathologist/ic_table_2class_exp2.xlsx"
data_raw = read_excel(path_array_raw,sheet = 1, col_names = TRUE, col_types = NULL, na = "", skip = 0)

data_all  = data_raw %>% select(2:32)
data_high = data_raw %>% select(2:12)
data_mid  = data_raw %>% select(13:22)
data_low  = data_raw %>% select(23:32)

print(path_array_raw)
print("data_all")
print("data_high")
print("data_mid")
print("data_low")
KappaM(data_all,  conf.level=0.95)   # Fleiss' Kappa and confidence intervals
KappaM(data_high, conf.level=0.95)   
KappaM(data_mid,  conf.level=0.95)   
KappaM(data_low,  conf.level=0.95)



## [3] Class 2, exp 3
path_array_raw="./IC_Score_Pathologist/ic_table_2class_exp3.xlsx"
data_raw = read_excel(path_array_raw,sheet = 1, col_names = TRUE, col_types = NULL, na = "", skip = 0)

data_all  = data_raw %>% select(2:32)
data_high = data_raw %>% select(2:12)
data_mid  = data_raw %>% select(13:22)
data_low  = data_raw %>% select(23:32)

print(path_array_raw)
print("data_all")
print("data_high")
print("data_mid")
print("data_low")
KappaM(data_all,  conf.level=0.95)   # Fleiss' Kappa and confidence intervals
KappaM(data_high, conf.level=0.95)   
KappaM(data_mid,  conf.level=0.95)   
KappaM(data_low,  conf.level=0.95)


sink() # finish to save log
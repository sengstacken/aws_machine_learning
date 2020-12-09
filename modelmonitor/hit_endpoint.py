import time
import numpy as np
import sagemaker
#from sagemaker.predictor import RealTimePredictor
import numpy as np

endpoint_name = 'nw-traffic-classification-xgb-ep-2020-08-31-22-29-39'

pred = sagemaker.predictor.Predictor(endpoint_name)
pred.content_type = 'text/csv'
pred.accept = 'text/csv'


dist_values1 = np.random.normal(1, 0.2, 1000)

# Tot Fwd Pkts -> set to float (expected integer) [second feature]
# Flow Duration -> set to empty (missing value) [third feature]
# Fwd Pkt Len Mean -> sampled from random normal distribution [nineth feature]

for i in range(150):
    
    # select random values to predict on
    sel = np.random.choice([1,2,3,4,5])
                     
    if sel == 1:
        artificial_values = "22,,40.3,0,0,0,0,0,{0},0.0,0,0,0.0,0.0,0.0,0.0368169318,54322832.0,0.0,54322832,54322832,54322832,54322832.0,0.0,\
54322832,54322832,0,0.0,0.0,0,0,0,0,0,0,40,0,0.0368169318,0.0,0,0,0.0,0.0,0.0,0,0,0,0,1,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,\
0.0,0.0,0.0,0.0,2,0,0,0,279,-1,0,20,0.0,0.0,0,0,0.0,0.0,0,0,23,2,2018,4,0,1,0"
        predict_string = artificial_values.format(str(np.random.choice(dist_values1)))
                     
    elif sel == 2:
        artificial_values = "80,76400,3,0,0,0,0,0,0.0,0.0,0,0,0.0,0.0,0.0,39.26701571,38200.0,40026.48646,66503,9897,76400,38200.0,40026.48646,66503,9897,0,0.0,0.0,0,0,0,0,0,0,96,0,39.26701571,0.0,0,0,0.0,0.0,0.0,0,0,0,0,1,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3,0,0,0,225,-1,0,32,0.0,0.0,0,0,0.0,0.0,0,0,16,2,{0},4,0,1,0"
        predict_string = artificial_values.format(str(np.random.choice([2018,2019,2020,2017,11])))
                     
    elif sel == 3:
        artificial_values = "80,22657,3,4,268,935,268,0,89.33333333,154.7298721,935,0,233.75,467.5,53096.17337,308.9552898,3776.166667,6245.364726,15141,2,15472,7736.0,10472.25143,15141,{0},22652,7550.666667,7729.022728,15469,26,0,0,0,0,72,92,132.4094099,176.54587990000005,0,935,150.375,330.6158485,109306.8393,0,0,1,1,0,0,0,1,1.0,171.8571429,89.33333333,233.75,0.0,0.0,0.0,0.0,0.0,0.0,3,268,4,935,65535,219,1,20,0.0,0.0,0,0,0.0,0.0,0,0,21,2,2018,2,0,1,0"
        predict_string = artificial_values.format(str(np.random.choice([331,332,333,334,335,0.1])))
                     
    elif sel == 4:
        artificial_values = "21,1,1,1,0,0,0,0,0.0,0.0,0,0,0.0,0.0,0.0,{0},1.0,0.0,1,1,0,0.0,0.0,0,0,0,0.0,0.0,0,0,0,0,0,0,40,20,1000000.0,1000000.0,0,0,0.0,0.0,0.0,0,0,0,1,0,0,0,0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1,0,1,0,26883,0,0,40,0.0,0.0,0,0,0.0,0.0,0,0,14,2,2018,2,0,1,0"
        predict_string = artificial_values.format(str(np.random.choice([2000000.0,1000000.0,1500000.0,3000000.0,2500000.0])))
                     
    else:
        artificial_values = "8080,12603,3,4,326,129,326,0,108.6666667,188.2161878,112,0,32.25,53.7672453,36102.51527,555.4233119,{0},4732.216679,11754,21,512,256.0,284.256926,457,55,12167,4055.666667,6669.132052,11754,36,0,0,0,0,72,92,238.0385622,317.3847497,0,326,56.875,115.4066568,13318.69643,0,0,1,1,0,0,0,1,1.0,65.0,108.6666667,32.25,0.0,0.0,0.0,0.0,0.0,0.0,3,326,4,129,8192,219,1,20,0.0,0.0,0,0,0.0,0.0,0,0,3,2,2018,5,0,1,0"
        predict_string = artificial_values.format(str(np.random.choice([2100.5,2101,'a',2000.,2150.,1900.])))
    
    print(predict_string)
    
    try:
        pred.predict(predict_string)
        print('Executed {0} inferences.'.format(i))
    except:
        print('error in prediction')
                     
    time.sleep(15*60) # sleep for 15 min

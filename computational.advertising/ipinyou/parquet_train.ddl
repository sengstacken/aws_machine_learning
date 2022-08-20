create table train 
with (
  external_location = 's3://sagemaker-us-east-1-431615879134/ipinyou-tf/data/parquet_train/',
  format = 'PARQUET'
)
as select * from ipenyou.ipenyoucsvtrain
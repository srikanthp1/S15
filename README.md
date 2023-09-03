# TRANSFORMER with encoder and decoder arc

## About 

* code for training is inside pylera15 folder 
* seeding everything using deterministic flag in trainer 
* model_train.py holds the training code with model code in pyt_model.py and model.py
* using batch_size 16 for optimized training time 
* started loss at about 6 
* end loss is close to 4
* 512 -> 2048 -> 512 in feed forward network (squeeze and expand)
* 8 heads with 6 repeatations 

## code 

* using pytorch lightning for writing code 
* weight initialization with xavier uniform distribution 
* logs and training code in S15.ipynb file


from sklearn.cluster import KMeans  
import pandas as pd
import time

# Add names to the data just for further manipulations
names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'result']

# import our dataset from the csv file and assign the given names
data_set = pd.read_csv('DataOk.csv', names = names)

# get dataset records count
ds_size = data_set.shape[0]

# set the amount of data to be predicted
prediction = 100

# get the number of data for the training
training = ds_size - prediction

# get data to for the training in a new array. we remove the result tab because it's not used for the training
training_ds = data_set.drop(columns = 'result').drop(data_set.index[training - prediction:ds_size - prediction ]).values

# data used as prediction input to get predicted output
predicted_input = data_set.drop(columns = 'result').drop(data_set.index[0:training]).values
 
# data output from the predicted input above
predicted_output = data_set.drop(columns =['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']).drop(data_set.index[0:training]).values

# Kmeans computation. start timer to track computation time
time_start = time.time()

# Define kmeans cluster number
k = 6

# create kmeans cluster object
kmeans = KMeans(n_clusters = k, init='k-means++', max_iter=300, n_init=10, random_state=0)

# apply training on training dataset previously prepared
train_result = kmeans.fit(training_ds)

# stop timer
time_end = time.time()
print("Training time : {:.3f} ms".format((time_end - time_start) * 1000))

# prediction computation.
time_start = time.time()

# Now based on that training, lets try to predict
prediction_output = kmeans.predict(predicted_input)

# stop timer
time_end = time.time()

# let's get the percentage of right prediction
match = 0
for i in range(prediction):
    if (predicted_output[i] == prediction_output[i]):
        match +=1
    

# get percentage value
percentage = match / prediction
print("\nPrediction time : {:.3f} ms".format((time_end - time_start) * 1000))
print("\nPrediction match : ", percentage * 100)


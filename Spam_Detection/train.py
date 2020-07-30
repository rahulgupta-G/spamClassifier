import naive_bayes
import pickle
from naive_bayes import SpamClassifier, trainData, testData

print("Training Data started")

sc_tf_idf = SpamClassifier(trainData, 'tf-idf')
sc_tf_idf.train()
preds_tf_idf = sc_tf_idf.predict(testData['message'])

print("Training Data Done!")

#Writing model to hdf5 file

print("Writing Data Started")

#This is the link of the hdf5 file inside which you will be writing the model
with open("Path_of_the_file_to_open_model/model.hdf5", "wb") as f:
    pickle.dump(sc_tf_idf, f)

print("Writing Data Done!")

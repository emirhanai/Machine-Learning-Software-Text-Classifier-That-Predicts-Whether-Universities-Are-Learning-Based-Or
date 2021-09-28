#import of libraries for software...
import pandas as pd
import numpy as np
from sklearn.model_selection import *

#Data Reading :)
df = pd.read_csv('university_learning.csv')

#Creating of Fetarues
X = np.array(
    df.iloc
             [:,
             0:1
             ]
)

#print(X)
#Creating of Labels
y = np.array(
    df.iloc
             [:,
             1:2
             ]
             )

#print(type(y))

#import of CountVectorizer function
from sklearn.feature_extraction.text import CountVectorizer

#numpy array to string is function running :)
x_list = np.array2string(X,separator=",")

#string to replace!
xx = x_list.replace("[","")
xxx = xx.replace("]","")
xxxx = xxx.replace("\n","").lower()

#print(xxxx)

#string to list!
xxxxx = xxxx.split(",")

#print(len(xxxxx))

#numpy array to string ;)
y_list = np.array2string(y,separator=",")

#replace of things ing string :)
yy = y_list.replace("[","")
yyy = yy.replace("]","")
yyyy = yyy.replace("\n","").lower()

#print(yyyy)

#string to list!
yyyyy = np.array(yyyy.split(","))

#print(len(yyyyy))

#print(x_list)

#CountVectorizer function is loading.
change_detection = CountVectorizer()

#fit and transform is feature list

change_detection.fit_transform(xxxxx)

change_x = change_detection.fit_transform(xxxxx)

#import is the TfidTransformer Function for feature and label
from sklearn.feature_extraction.text import TfidfTransformer

#Loading is TfidTransformer function :)
change_tfid = TfidfTransformer()

#CountVectorizer type Feature to TfidTransformer type :)
change_xx = change_tfid.fit_transform(change_x)

#print(type(change_xx))

#scipy.sparse.csr.csr_matrix to numpy array
change_x_toarray = change_xx.toarray()

#CountVectorizer function loading...
y_change_function = CountVectorizer()

#fit and transform to label in CountVectorizer function
change_of_y = y_change_function.fit_transform(yyyyy)

#print(type(change_of_y))

#scipy.sparse.csr.csr_matrix to type of numpy array and reshape to label
change_of_y_toarray = (change_of_y.toarray()).reshape(22,-1)

#reshape of fature array :)
change_array = np.array(change_x_toarray).reshape(22,-1)

#print of feature array and label array..
print(change_array.shape)
print(change_of_y_toarray.shape)

#for loop creating for regularly software..
for i in np.arange(1,2,1):

    #Creating (separate) of Train and Test datasets..
    X_train, X_test, y_train, y_test = train_test_split(change_array, change_of_y_toarray,
                                                        test_size=0.136, shuffle=True, stratify=None,
                                                        random_state=33)

    #import of model
    from sklearn.ensemble import ExtraTreesClassifier

    #creating of model
    model = ExtraTreesClassifier(criterion="gini",n_estimators=1,random_state=1,)

    #print(X_train.shape)
    #print(y_train.shape)
    #print(y_train)

    #fitting is dataset in model
    model.fit(X_train,y_train)

    #prediction personal comment!
    prediction = ["Universities are not for learning."
                  "Learning is about self-development."
                  "Universities restrict people in learning and prevent them from developing in world of science."
                  "In this case, we should act knowing that information is free."
                  "We can train ourselves in learning knowledge."
                  "It's easy and free to find information."]

    #prediction comment (list type) to scipy.sparse.csr.csr_matrix type
    predict_change = change_detection.fit_transform(prediction)

    #(predict_change.shape[1])
    try:
        while True:
            if predict_change.shape[1] == 37:
                
                #scipy.sparse.csr.csr_matrix prediction type to numpy array type
                predict_to_array = np.array(predict_change)

                #prediction is the data!
                prediction_system = model.predict(predict_change)

                #Creating of the For loop. Because class name is the None :))
                
                for i in prediction_system:
            
                    if prediction_system == np.array(1):
            
                        print(f"Yes! Don't need to Univeristy for learning, {yyyyy[0].upper()}")
                    elif prediction_system == np.array(0):
                        
                        print(f"Yes! Need to Univeristy for learning, {yyyyy[1].upper()}")

                #Different and exactly class name
                class_name_two = ["not for learning","for learning"]

                #Creating for loop because prediction class...
                for information, name in zip(class_name_two, prediction_system):
        
                    print('%r => %s' % (information, xxxxx[name]))

                    # Accuracy Score!
                    print("Accuracy Score: ", np.mean(prediction_system == y_test) * 100, "x: ", i)
                    
                break
            else:
                print("Please, try again!")
    except:
        print("Please, try again!")

    #feature names are loading..
    feature_names = df.iloc[:,0:1]

    #import is the export graphviz function at tree in sklearn :))
    from sklearn.tree import export_graphviz

    #tree classification creating of the visulation or graph!
    def save_decision_trees_as_dot(model, iteration):
        file_name = open("emirhan_project" + str(iteration) + ".dot", 'w')
        dot_data = export_graphviz(
            model,
            out_file=file_name,
            class_names=["for learning","not for learning"],
            rounded=True,
            proportion=False,
            precision=2,
            filled=True, )
        file_name.close()
        print("Extra Trees in forest :) {} saved as dot file".format(iteration + 1))

    #Crate of tree graph about of text classification :))
    #for i in range(len(model.estimators_)):
    #    save_decision_trees_as_dot(model.estimators_[i], i)
    #    print(i)

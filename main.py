import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 


music_data= pd.read_csv("music.csv")
X = music_data.drop(columns=["genre"])
Y = music_data["genre"]
X_tain,X_test,y_tyrain, Y_test =train_test_split(X,Y,test_size=0.2) #setting 20% of values to testing

model = DecisionTreeClassifier()

model.fit(X.values,Y)
predictions = model.predict([ [21,1] , [ 22,0]])
predictions

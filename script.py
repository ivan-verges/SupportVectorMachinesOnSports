import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

#Creates a Subplot graph to show data
fig, ax = plt.subplots()

#Updates data in the Jose Altuve dataset transforming the Type column values from String to Numbers
jose_altuve["type"] = jose_altuve["type"].map({"S" : 1, "B" : 0})

#Removes the NA values from the dataset
jose_altuve = jose_altuve.dropna(subset = ["plate_x", "plate_z", "type"])

#Prepare the graph to show the data
plt.scatter(jose_altuve["plate_x"], jose_altuve["plate_z"], jose_altuve["type"], cmap = plt.cm.coolwarm, alpha = 0.5)

#Loads the Training and Validation data from Jose Altuve dataset
training_set, validation_set = train_test_split(jose_altuve, random_state = 1)

#Creates a Support Vector Classifies (Support Vector Machine) to train, using the Radial Basis Function Kernel (RBF), a Gamma (Spread of the Desicion Region) of 9 and C (Miss Classification Penalty) of 8
classifier = SVC(kernel = "rbf", gamma = 9, C = 8)

#Removes unused Features and Labels and just leaves the desired ones
training_data = training_set[["plate_x", "plate_z"]]
training_labels = training_set["type"]

#Trains the model with Training data
classifier.fit(training_data, training_labels)

#Uses the Draw Boundaries into th Sub-Plot element we created before
draw_boundary(ax, classifier)

#Set region Limits to the Plot Area
ax.set_ylim(-2, 6)
ax.set_xlim(-3, 3)

#Show the plot
plt.show()
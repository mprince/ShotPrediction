The goal of this exercise was predicts the likelihood that a given basketball shot will go in. Play by play data is provided through http://www.basketballgeek.com/data/  for the the 2006-2010 seasons. 

So given a set of data, a system has to be trained in order to predict the correct results up to a certain level of confidence. So given a shot at point (x,y), the system should to see if the shot was made or missed. I thought of implementing a neural network, a machine learning tool inspired by the brain(http://en.wikipedia.org/wiki/Artificial_neural_network). So in this repo will be included the implementation of a simple neural network using fully-connected layers, and trained using the backpropagation algorithm. This means that for each shot, the following will be done:
* Use the shot coordinates to set the activations of your input layer
* Propagate the activations up to the top layer
* Compare the top layer activation with the result of the shot
* Backpropagate the error (if any) down the network to adjust weights

The parameters for the learning algorithm are as follow:
inputSize: would be equal to 2 as the number of coordinates, one for x and one for y
outputSize: would be equal to 2 as the result of the shot, 0 for missed and 1 for made
hiddenSize: number of neurons or size of the hidden layer
learn: learning rate
epochs: number of training sessions

In this code I have trained the system through 10 different games from the 2006 season. Then I picked up a game and tested the accuracy of the system against it. So the program will run through these steps and output in console the number of shots predicted correctly, the total number of shots tested and the accuracy. To run just make sure you have the csv files in the same folder as the code and run the following commands:
javac ShotPrediction.java
java Shotprediction

Unfortunately as of now I am not getting a satisfactory prediction rate (~70%). A possible I have yet not find the convenient parametes (hiddenSize, learn, epochs). I also would like after that maybe add a second hidden layer that would increase accuracy even though at the cost of performance since more calculations have to be made. Also it is important that I am only predicting based on the coordinates of the shot. Given the data I have also the type shot that is being made (3pt, jump, hook). If I could quantify this paramater and use it in my input layer as to help me with the prediction.
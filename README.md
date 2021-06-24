# Deep-Learning-for-the-MNIST-Data-Set
I have implemented, in MATLAB, an artificial neural network (ANN) for handwritten digit recognition using the MNIST data set. This deep learning project involved training the network, tuning hyperparameters, testing the network whilst simultaneously documenting the steps taken to produce a report and create a recorded audio-video presentation showcasing the ANN. 

Run this code snippet to generate, Confusion matrix for calculus based backprop using the smaller dataset,

ann1923114(100, .01, 1, 2,103,57,31,1,2,3,9)

To run the test place the code and the dataset in the same folder.
Key reminder: "There 5 classes in the confusion matrix, the first four are digits and the last one is the none class."


Going through the handwritten digit recognition ANN I made in under 5 minutes. (speed-run); https://youtu.be/AcSmXXuit6k



Subtitles for video:

Here, I have written a function that takes 11 user-defined arguments; number of epochs, learning rate, data: for selecting the large or small dataset, bp: for selecting desired backpropagation schemes, U, V, W: numbers of hidden layers neurons and A, B, C, D: the desired digits to classify and the function returns both predicted and ground-truth values of the training and testing dataset. You can see it's represented in line 1 within the code.
I have also introduced multiple checking schemes to check whether the user gives appropriate data or not.  You can see this between line 29 and 37 in the code. 
According to user-defined data (1 or 2), the training data set is loaded. You can see this between line 41 and 46. The image of the digit has been normalized between 0 to 1 for faster training of the model. You can see this in line 48. Here the first four classes represent the desired digit class respectably where the last class represents the None class meaning the image won't belong to any of the desired classes. You can see this between line 49 and 65 in the code.

The proposed network has an input layer of 784 nodes as the image resolution is 28X28 with 5 output nodes and 3 hidden layers each with user-defined neuron numbers. You can see this between lines 68 and 69 in the code. Then the weight of the successive connecting nodes have been initialised as a random value ranging from -0.5 to 0.5 and zeros are allocated as an initial bias. You can see this in the first page of the report and in line 71-74 in the code. The sigmoid and softmax activation function has been defined, this is used in the hidden layers and output layers respectably. You can see this in line 76 of the code to 79 and in the fifth page of the report.

Then the training phase starts. In this phase, the network tries to optimize the weights and biases with the help of forward propagation and backpropagation.
The first for loop functions between the user-defined number of epochs which is represented as N_ep in line 87 of the Matlab code. In each epoch there is a randomly selected image from the training set that has been fed to the network for training. You can see this between line 88 and 93 in the code. The input image has been successively transmitted between neurons with the applied weights along with the biases. The weighted sum of inputs with biasing has been passed through the activation function, this determines whether to activate the neuron or not.  You can see this between 96 and 99 in the code. In the last layer, the applied softmax layers predicted the probability of an input belonging to each class. You can see this in line 101. 

 In the backpropagation part, I applied both unscaled heuristic backpropagation and calculus-based backpropagation. The error in the final layer was (stutter) this as you can see here in the report. We then fed those errors to the previous layers by multiplying that to the layer weights, as shown in the report here. The backpropagation coefficient for the last layer was then set to the error in the last layer, i.e. S5=-e5.  You can see this in line 110 in the Matlab code. 
To find the other backpropagation coefficient, Sn, I found the derivatives of the inactivated weights(nn) and then put them into a diagonal matrix called An. You can see this in the second page of the report and you can see this in the Matlab code between line 103 and 106. 
These errors were then multiplied with the An values to find the final backpropagation coefficient. Where the bp coefficient is equal to -2 * Diagonal of the activation derivative * error. 
Now for the calculus-based backpropagation, serial multiplication was performed to find the backpropagation coefficient of the previous layer using the function: You can see this in the second page of the report where the bp coefficient is equal to the Diagonal of the activation derivative * weight of the forward layer  * bp coefficient of the forward layer. You can see this represented in the Matlab code between line 119 and 121. 
This backpropagation coefficient is then multiplied with activated node values and a learning rate that is subtracted from the layer weights. The same is done for the biases. You can see this done between line 129 and 132 in the Matlab code. 
Finally, for every output value, SoftMax is applied on the complete forward propagation layer. The line is shown in the MATLAB code in line 137. This line is extremely important as it connects the input with the output, as you can see this line takes in all the weights consecutively and passes them from the input layer through the hidden layers to the output layer. The term “sig2” refers to the sigmoid activation function, whereas the term “sig5” refers to the softmax activation function.  
For plotting the cross-entropy error I used the following function seen in the MATLAB code in line 138. This line calculates the cross-entropy value, as the cost function and then plots the value in a subplot.  

Now let us run the code for the smaller data set and see how it works.

So now we're going to see 3 windows pop up. We're going to see figure one which is essentially the training data set and testing data set running at the same time and then now we're gonna see the (stutter) two confusion matrices.
The one on the left is essentially going to be the training (stutter) data set confusion matrix. Essentially where the predicted output is the output class and the target class is the true output. You can see that we have a training accuracy of 100%. 
Now on the right, we can see the confusion matrix with the testing uh data set. Now essentially here we have an accuracy of only 80% and of course, now we're going to see that there are a few numbers on here. Of course the diagonal is referring to the true positives where essentially that means that there were five images that were predicted to be in the fifth class and were actually from the fifth class and there was for example one image that was predicted to be in the fifth class but turned out to be in the second class

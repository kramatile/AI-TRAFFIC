a.1. while reading documentation of keras i saw that they always standarized the data before deploying it so we will use in the neural network the following code line:
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))

a.2. to reduce the size i noticed that we should use one or multiple layers of convolution and pooling (max pooling in my case): example i saw : (https://www.tensorflow.org/tutorials/images/classification?hl=fr)
	layers.Conv2D
	layers.MaxPooling2D()
	my intuition tells me to use three of those so thats what ill do :).

a.3. at the end we will have a dense layer with 43 outputs : with an activation = softmax wich will give a probability distribution i guess but will see... 
	layers.Dense(num_classes)

a.4 ill also consider adding dropouts so we dont rely on any particular node or neuron:

a.5 add a dense layer before the last layer : it will have 128 neuron cause i saw this number comming back a lot of times ! 

a.6 in the comiler i chose accuracy 

# first idea ! 
------------------------------------------------------------------------------------------------------------------------------------------------------------
second try 
b.1 the overall accuracy was low so ill try to reduce the number of convolution layers to 1 
b.2 change the activation methode to softmax in the final dense layer

third try add a convolution layer : 

welll good enough :)
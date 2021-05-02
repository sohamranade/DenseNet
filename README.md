# DenseNet
This is a denseNet implementation using keras and tensorflow. 



# this is a densenet implementation based on the paper https://arxiv.org/abs/1608.06993 using keras and tensorflow.
#inspiration regarding the implementation was also taken from https://www.youtube.com/watch?v=QKtoh9FJIWQ&t=1747s and https://github.com/cmasch/densenet



#arguments: 
#in_shape= shape of the input
#nb_classes= number of classes in the output
#nb_denseblocks= number of dense blocks
#nb_layers= number of layers in a block
#growth_rate= growth rate of the architecture
#depth= depth of the neural network
#compression= compression ratio (0-1)
#bottleneck= (bool) Wether to include bottleneck in the system or not.
#dropout= (0-1) the dropout to be given
#weight_decay= weight decay to be applied to the model

#returns created model with all the parameters.

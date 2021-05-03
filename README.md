# DenseNet using KERAS and Tensorflow
This is a denseNet implementation based on the paper https://arxiv.org/abs/1608.06993 using keras and tensorflow.


Inspiration regarding the implementation was also taken from https://www.youtube.com/watch?v=QKtoh9FJIWQ&t=1747s and https://github.com/cmasch/densenet

arguments:</br>
in_shape= shape of the input<br/>
nb_classes= number of classes in the output<br/>
nb_denseblocks= number of dense blocks<br/>
nb_layers= number of layers in a block<br/>
growth_rate= growth rate of the architecture<br/>
depth= depth of the neural network<br/>
compression= compression ratio (0-1)<br/>
bottleneck= (bool) Wether to include bottleneck in the system or not.<br/>
dropout= (0-1) the dropout to be given<br/>
weight_decay= weight decay to be applied to the model <br/>

returns created model with all the parameters. <br/>

Convolutional Neural Networks (CNNs) are a powerful class of deep learning models designed specifically for processing grid-like data, such as images. They've revolutionized computer vision tasks, from image classification to object detection. In this post, we'll dive into the fundamental concepts of CNNs and how they work.

One significant challenge with regular neural networks is the fact that with the size of image, the number of parameters scales very very quickly. Imagine the huge complexity of the network if you have an image of say 200x200x3 which would mean we have 120k weights (this is all taken from the cs231n course. I have left a link to it at the bottom of the page). 
What is the solution ? Pass in the image <b>as an image</b>. This may sound confusing at first but the idea is simple, a regular neural network can take in only a 1D tensor so all the images are first flattened before being passed in. The advantage of passing an image as an image is that you preserve its overall 2D/3D structure. This 3D structure preservation also ensures that you can utilize the <b>spatial information</b> in images. 

## Structure of CNN
Now that we understand the basic idea, let's look at how a CNN is structured
This section will make things a whole lot clear to understand (hopefully). 
Below is a rough sketch (made by yours truly) of a regular neural network. We have an input layer, then few hidden layers and lastly the output layer. Simple and straightforward right ? Now imagine 120,000 neurons in the input layer (truly frightening thought)
![[neural_net_sketch.png]]

Now let me show you what a ConvNet looks like (rough sketch btw)
![[convnet.jpeg]]

(ok so i wasn't able to make a rough sketch on my own, so i got this diagram from <a href="https://editor.analyticsvidhya.com/uploads/90650dnn2.jpeg">here</a> that shows it a lot better)

**Recap:** CNNs consist of specialized layers that work together to process and understand visual information, dramatically reducing the number of parameters compared to regular neural networks.

Right now you are extremely confused as to what you are looking at so lets break it down one by one
### Input
The input will typically have dimensions of [height, width, channels] so for example an image of size 32x32x3. 
### Convolutional Layer

Here you will take a filter of the same <b>depth</b> / channels as the input image but a smaller spatial area (i.e, smaller height, width) and we are going to do a <b>dot product</b> between the filter and the small chunk of the image that the filter's spatial area covers while sliding over the whole image. So for example, if we have a filter of size 5x5x3, we will do 75 dot products + bias. You can think of it as doing element-wise multiplication at each area. 
To make it simpler to understand, we 'stretch out' or flatten the filter and then perform dot-product. 

Earlier I mentioned that the filter 'slides' over the image. Now what does this exactly mean ? 
Think of the filter as a spotlight moving across a stage, highlighting different parts of the performance.
First off the interval at which the filter is slid/moves utilizes a concept called 'stride'. 
This can also be considered a hyper-parameter. Now how do you decide what is a good stride ? Usually this is done through experimentation as for different use cases different strides are found to be ideal. But one thing to remember though, is that if you go <b>too far</b> you are going to end up with your filter going <b>outside</b> the image. This means that a part of the spatial area covered by the filter may be outside the image and so we can use this nice little formula to determine the size of the activation map: 

$$output size = (N-F)/stride + 1$$
Where:

$N$ : Dimensions of input 
$F$ : Dimensions of filter
$stride$ : stride

So for example, lets say I have an input of dimension 7 (i.e width=7, height = 7) and filter of size 3 along with stride of 2. Is this viable ? lets check by applying the formula: 
$outputsize = (7-3)/2 + 1 = 3$

Yep this is viable. Why ? Because we get a size which is a whole number (it is pretty obvious why we would want our activation map to have a whole number size). 

Lets try another example. lets say I have an input of dimension 7 (i.e width=7, height = 7) and filter of size 3 along with stride of 5. Is this viable ? lets check by applying the formula: 
$outputsize = (7-3)/5 + 1 = 1.8$
ALERT. CANNOT BE USED. 

Ok lets say maybe you want a stride that, using the equation does not work out. 
Zero-padding can be a potential solution. It helps maintain the size of output and apply filter to the <b>corner regions of the image</b>
In order to understand the amount of zero padding you need to do in order to maintain the spatial area for the chosen filter you can use the formula: $(F-1)/2$ 

Now, the depth of the output from convolutional layer depends on the <b>number of filters</b>
Lets see an example here: 
Input Volume: 32x32x2
10 5x5 filters with stride 1 and pad 2

<span style="color: red">What is the output volume size ?</span>
since we are applying padding , this will increase each dimension by 2 so the new dimension is $32 + 2*2$ which is equal to 36. Next we are applying a filter which will mean reduction of the size by the filter dimension so $36-5 = 32$ then finally, divide by 1+1 to get 32. The depth as I mentioned earlier will be equal to the number of filters which in this case is 10. 
<span style="color: red">What is the number of parameters ? (in total), assuming that the bias is 1</span>
each filter has $5*5*3 + 1 = 76$ params and so 10 of them will have 760. 

I found a great visualization of this layer <a href="https://openlearninglibrary.mit.edu/assets/courseware/v1/cda92ed2c6672271916e8cb8974af568/asset-v1:MITx+6.036+1T2019+type@asset+block/notes_conv_nets_slides.pdf">here</a>
I will be pasting a ss from one of those slides
![[conv_layer.png]]
![[convlayer_2.png]]
above image taken from this paper <a href="https://link.springer.com/article/10.1007/s10462-024-10721-6">link</a>

**Recap:** To sum up, convolution helps us detect features in images using filters that slide across the input.

#### Implementation of Conv Layer in Python
I am going to implement a simple conv layer here to demonstrate the logic and mathematical concepts we discussed earlier. To start with, I am going to put a cheat sheet here to help yall much better understand how exactly numpy works 

[[Numpy Cheat Sheet]]

So lets start with the forward pass. The first step I am going to take here is checking for zero-padding. As we discussed earlier, zero-padding will add a few extra rows and columns to your input in order to be able to capture features from the edges of the images. 



### Pooling Layer
Essentially, the pooling layer downsamples the input while maintaining most of the important features/aspects of the image. 

Accepts input of size WxHxD (WidthxHeightxDepth)
Hyper params: 
- F (size of filter. How much spatial area to cover)
- S (stride)
Output: 
- $W_2 = (W-F)/S+1$
- $H_2 = (H-F)/S+1$
- $D_2=D_1$
#### Max pooling
We use a filter of size F and within the spatial area of the filter, we choose the max value within the region. The visualization below does a brilliant job of simplifying the concept.  

![[max-pooling.png]]
##### Effects/Reasons to use
When the dimensions of the input get reduced, the number of input values fall which in turn also reduce the number of parameters and that in turn reduces number of computationn which in turn also prevents overfitting 

I think max-pooling is the simplest layer to implement given that its concept is straightforward. Obviously if you are not using a library to implement it you will need to be extra careful of the shapes that are being input in the forward pass function of the layer. 
Otherwise if you use something like PyTorch, the implementation is simple and you can find loads of guides to follow (some of my favourite ones are : <a href="https://www.learnpytorch.io">learnpytorch</a> , <a href="https://pytorch.org/tutorials/beginner/basics/intro.html">pytorchdocs</a>)

This is the most common type of pooling layers used. Others are :
#### Global Pooling
Reduces each feature map to a single value, often by taking the average or max across the entire map. Commonly used at the end of networks to reduce spatial dimensions to 1x1.
#### Average Pooling
Works similar to max pooling except in each spatial area covered by the filter, you take the average of all the values within the area instead of the max. 

To get even more detailed information as well as a detailed implementation of this layer you can check out this <a href="https://blog.paperspace.com/pooling-in-convolutional-neural-networks/"> link </a>

**Recap:** Pooling layers downsample the feature maps, reducing computational load while preserving important information.
### Flattening Layer

Extremely crucial layer and it is placed between conv/pooling layer and the fully connected layer. Why is it crucial ? Because the output from conv/pooling layer is a 3D output and a FC Layer does <span style="color:red">NOT</span> like that. It can only take in a 1D input so we need to squish the 3D output into 1D. 

![[flattening_layer.png]]
Source: <a href="https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-3-flattening">link</a>

Input: 
- image of dimensions WxHxD

Output: 
$output = reshape(input, [1, W * H * D])$
so basically a 1D vector of size $[1 \times (W * H * D)]$

For example:
If the input is $4 \times 4 \times 3$, the flattened output will be a vector of size $[1 \times 48]$

<b>Note</b>
Always use the flattening layer just before the FC Layer. 
(Something else to note, not every dl library requires you to declare the Flattening Layer or add it explicitly. Some have it as part of the FC Layer itself)

**Recap:** The flattening layer transforms the 3D output from convolutional and pooling layers into a 1D vector for the fully connected layer.

#### Fully Connected Layer
Aaaand we have arrived to the last layer of the CNN, the fully connected layer. 
This is where the 'brain' of the cnn processes all the features obtained so far. To keep it simple and easy to understand I will keep this part to the point: 

- We initialize weights and biases in the FC Layer. Every neuron in the FC layer connects to every feature in the input.
- Each connection has a 'weight' and every neuron has a 'bias'. This follows the same concept as a regular neural network
- It then calculates the neuron's decision using the beautiful formula $z = Wx + b$ where W is the $W$ is the weight and $b$ is the bias. In practical, this operation is performed for all the connections in parallel (as in every calculation is done at the same time)
- After calculation of $z$ you need to apply an activation function. Why ? because an activation function will squish the value between a specfied range and this helps to much more easily understand and interpret the output. Most common one (for multiclass classification: Softmax : $f(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$  . Here the output sums to 1 )
- The output from the activation function will be the probability of the input being a specific class. The output neuron with the highest probability is the predicted class.

**Recap:** Fully connected layers take the flattened input and perform high-level reasoning, ultimately producing the network's final output.
## Next steps
Well what I described here is the first half of what goes on in a CNN. This process is known as the forward propagation or the forward pass. The other half is known as the backpropagation. Given how huge of a topic backprop is, I will be making a separate blog/article on it which will come out in a few weeks. To summarize backprop, it is basically the neural network learning from its mistakes. 

To recap the entire forward prop: 
- Convolutional Layer: We use this layer to learn local features and spatial relationships. 
- Pooling Layer: Downsample the image in order to reduce dimensions and computations but preserve important features
- Flattening Layer: Flatten the output from previous layer and get it ready for FC Layer. This is usually the conversion from 3D to 1D. 
- FC Layer: Connect all input features with all the neurons in the layer and get the weighted sum of inputs. We use that to get the probabilities for all classes and class with the highest probability is the predicted class




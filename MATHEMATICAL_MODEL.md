# XOR from Scratch: The Mathematics of a 2 → 2 → 1 Neural Network

This page develops, step by step, a small neural network capable of learning the **XOR** function.

The goal is not to rely on a machine-learning library, but to understand what the network is doing both geometrically and mathematically. In particular, we will see how neurons, weights, biases, activation functions, loss minimization, gradient descent, and backpropagation work together.

---

## 1. The XOR problem

The XOR function is defined by the following table:

| $x_1$ | $x_2$ | $y$ |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

In the $(x_1,x_2)$ plane, the two points with output $1$ lie at opposite corners of the unit square. The same is true of the two points with output $0$.

Consequently, no single straight line can separate the two classes. A linear neuron is therefore not sufficient to solve XOR.

The purpose of the hidden layer will be to transform the original points into a new space in which the two classes *can* be separated by a line.

---

## 2. Network architecture

We use a neural network with architecture

$$
2 \longrightarrow 2 \longrightarrow 1.
$$

It consists of:

- two inputs, $(x_1,x_2)$;
- two hidden neurons, $(h_1,h_2)$;
- one output neuron, $\hat y$.

The hidden neurons compute

$$
h_1=\sigma(w_{11}x_1+w_{12}x_2+c_1),
$$

$$
h_2=\sigma(w_{21}x_1+w_{22}x_2+c_2),
$$

while the output neuron computes

$$
\hat y=\sigma(v_1h_1+v_2h_2+c_3).
$$

The complete vector of trainable parameters is

$$
\theta=
(w_{11},w_{12},w_{21},w_{22},c_1,c_2,v_1,v_2,c_3).
$$

The network therefore has nine trainable parameters.

---

## 3. The geometric meaning of a hidden neuron

Before applying the activation function, the first hidden neuron forms the linear combination

$$
z_1=w_{11}x_1+w_{12}x_2+c_1.
$$

The equation

$$
z_1=0,
$$

or equivalently

$$
w_{11}x_1+w_{12}x_2+c_1=0,
$$

defines a straight line in the $(x_1,x_2)$ plane. Similarly, the second hidden neuron defines the line

$$
w_{21}x_1+w_{22}x_2+c_2=0.
$$

<img src="xor_input_space.png" width="500" alt="XOR points and hidden-neuron threshold lines">

The weights determine the orientation of each line, while the bias allows the line to move away from the origin. Without a bias, every threshold line would be forced to pass through the origin.

It is important to distinguish the lines from the neuron outputs. The lines are the sets of points for which the arguments of the sigmoids are zero. The quantities $h_1$ and $h_2$, on the other hand, are the values obtained *after* the activation function has been applied.

---

## 4. The sigmoid activation function

The activation function used here is the sigmoid

$$
\sigma(a)=\frac{1}{1+e^{-a}}.
$$

Its behavior can be summarized as follows:

$$
a\ll0 \quad\Longrightarrow\quad \sigma(a)\approx0,
$$

$$
a=0 \quad\Longrightarrow\quad \sigma(a)=0.5,
$$

$$
a\gg0 \quad\Longrightarrow\quad \sigma(a)\approx1.
$$

Consider a generic hidden neuron

$$
h=\sigma(w_1x_1+w_2x_2+c).
$$

Its threshold line

$$
w_1x_1+w_2x_2+c=0
$$

divides the plane into two half-planes. On the side where

$$
w_1x_1+w_2x_2+c<0,
$$

the value of $h$ tends toward $0$. On the other side, where

$$
w_1x_1+w_2x_2+c>0,
$$

it tends toward $1$.

The sigmoid therefore behaves like a **soft threshold**. Unlike a discontinuous step function, it changes smoothly and usually does not return exactly $0$ or $1$. Nevertheless, when its argument has a sufficiently large magnitude, its output becomes extremely close to one of these values.

---

## 5. From lines to new coordinates

Each hidden neuron measures the position of the input relative to its own threshold line. Together, the two neurons transform the original coordinates according to

$$
(x_1,x_2)\longmapsto(h_1,h_2).
$$

This is the key role of the hidden layer: it does not directly decide the final class. Instead, it constructs a new representation of the input.

Because the sigmoid is nonlinear, this transformation can rearrange the data in a way that no purely linear transformation could. The output neuron then performs a linear separation in the new $(h_1,h_2)$ space.

---

## 6. An intuitive geometric solution to XOR

To see how this works, consider the following two hidden neurons:

$$
h_1=\sigma(10x_1+10x_2-5),
$$

$$
h_2=\sigma(10x_1+10x_2-15).
$$

Their threshold lines are

$$
10x_1+10x_2-5=0
\quad\Longleftrightarrow\quad
x_1+x_2=0.5
$$

and

$$
10x_1+10x_2-15=0
\quad\Longleftrightarrow\quad
x_1+x_2=1.5.
$$

These parallel lines divide the input plane into three regions:

- below the first line;
- between the two lines;
- above the second line.

The XOR points with output $1$ lie in the middle region, whereas the points with output $0$ lie in the two outer regions.

This is only one convenient and geometrically transparent solution. During training, the network is free to discover other parameter configurations that produce the same classification.

---

## 7. Computing the hidden coordinates

We can now calculate the values of $h_1$ and $h_2$ for each XOR input.

### Input $(0,0)$

$$
h_1=\sigma(-5)\approx0.0067,
$$

$$
h_2=\sigma(-15)\approx0.
$$

Therefore,

$$
(0,0)\longmapsto(h_1,h_2)\approx(0,0).
$$

### Input $(0,1)$

$$
h_1=\sigma(5)\approx0.9933,
$$

$$
h_2=\sigma(-5)\approx0.0067.
$$

Therefore,

$$
(0,1)\longmapsto(h_1,h_2)\approx(1,0).
$$

### Input $(1,0)$

By symmetry,

$$
(1,0)\longmapsto(h_1,h_2)\approx(1,0).
$$

### Input $(1,1)$

$$
h_1=\sigma(15)\approx1,
$$

$$
h_2=\sigma(5)\approx0.9933.
$$

Therefore,

$$
(1,1)\longmapsto(h_1,h_2)\approx(1,1).
$$

The complete transformation is approximately:

| Input $(x_1,x_2)$ | $h_1$ | $h_2$ | Target $y$ |
|:---:|:---:|:---:|:---:|
| $(0,0)$ | $0.0067$ | $\approx0$ | $0$ |
| $(0,1)$ | $0.9933$ | $0.0067$ | $1$ |
| $(1,0)$ | $0.9933$ | $0.0067$ | $1$ |
| $(1,1)$ | $\approx1$ | $0.9933$ | $0$ |

---

## 8. Interpreting $h_1$ and $h_2$

The hidden values are not yet the final prediction. They encode which thresholds the input has crossed:

$$
(h_1,h_2)\approx(0,0)
$$

means that the point lies below both thresholds;

$$
(h_1,h_2)\approx(1,0)
$$

means that it has crossed the first threshold but not the second;

$$
(h_1,h_2)\approx(1,1)
$$

means that it has crossed both thresholds.

The hidden layer has thus replaced the original coordinates with two new features that describe the point's position relative to the learned boundaries.

---

## 9. Linear separation in the hidden space

After the transformation, the four XOR inputs are arranged approximately as follows:

$$
(0,0)\longmapsto(0,0),
$$

$$
(0,1)\longmapsto(1,0),
$$

$$
(1,0)\longmapsto(1,0),
$$

$$
(1,1)\longmapsto(1,1).
$$

<img src="xor_hidden_space.png" width="500" alt="XOR points in the hidden space">

The two positive examples are now concentrated near $(1,0)$, while the negative examples lie near $(0,0)$ and $(1,1)$. In this transformed space, the classes are linearly separable.

For example, the line

$$
h_1-h_2-0.5=0
$$

separates the positive point $(1,0)$ from the two negative points. The output neuron can therefore behave approximately as

$$
\hat y=\sigma\left(K(h_1-h_2-0.5)\right),
$$

where $K>0$ is sufficiently large.

The complete geometric mechanism is

$$
\boxed{
(x_1,x_2)
\xrightarrow{\text{nonlinear hidden layer}}
(h_1,h_2)
\xrightarrow{\text{linear separation and sigmoid}}
\hat y
}.
$$

The hidden layer transforms the space; the output layer classifies the transformed points.

---

## 10. The output neuron

In general, the output neuron computes

$$
\hat y=\sigma(v_1h_1+v_2h_2+c_3).
$$

It does not receive $(x_1,x_2)$ directly: it sees only the features $(h_1,h_2)$ constructed by the hidden layer.

The two groups of parameters therefore have different roles:

- $(w_{ij},c_1,c_2)$ determine how the original input space is transformed;
- $(v_1,v_2,c_3)$ determine how the transformed points are classified.

All these parameters are learned together during training.

---

## 11. The loss function

To train the network, we need a numerical measure of the difference between the prediction $\hat y$ and the correct value $y$. A simple choice is the **mean squared error**:

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2.
$$

For the four XOR examples, this becomes

$$
\begin{aligned}
L(\theta)=\frac14\big[&\hat y(0,0;\theta)^2
+(\hat y(0,1;\theta)-1)^2\\
&+(\hat y(1,0;\theta)-1)^2
+\hat y(1,1;\theta)^2\big].
\end{aligned}
$$

Training is therefore the optimization problem

$$
\theta^\star=\arg\min_\theta L(\theta).
$$

The network is not explicitly instructed to place its threshold lines at $x_1+x_2=0.5$ and $x_1+x_2=1.5$. It receives only the examples and their correct outputs. The positions of the lines emerge from the adjustment of the parameters required to reduce the loss.

---

## 12. Gradient descent

Gradient descent reduces the loss by updating every parameter in the direction opposite to its derivative:

$$
p\leftarrow p-\eta\frac{\partial L}{\partial p},
$$

where $p$ denotes any weight or bias and $\eta>0$ is the learning rate.

Writing all parameters together gives

$$
\theta_{k+1}=\theta_k-\eta\nabla L(\theta_k).
$$

During training, this rule adjusts both the hidden-layer parameters

$$
w_{11},w_{12},w_{21},w_{22},c_1,c_2
$$

and the output-layer parameters

$$
v_1,v_2,c_3.
$$

Geometrically, updates to the hidden weights and biases rotate and translate the threshold lines. At the same time, updates to the output layer change the separating boundary in the hidden space.

---

## 13. Backpropagation

To understand how the required derivatives are computed, consider one training example and define

$$
L=\frac12(\hat y-y)^2.
$$

Introduce the pre-activation values

$$
z_1=w_{11}x_1+w_{12}x_2+c_1,
$$

$$
z_2=w_{21}x_1+w_{22}x_2+c_2,
$$

$$
z_3=v_1h_1+v_2h_2+c_3.
$$

Then

$$
h_1=\sigma(z_1),\qquad
h_2=\sigma(z_2),\qquad
\hat y=\sigma(z_3).
$$

The derivative of the sigmoid is

$$
\sigma'(a)=\sigma(a)(1-\sigma(a)).
$$

We begin at the output. By the chain rule,

$$
\delta_3
=\frac{\partial L}{\partial z_3}
=(\hat y-y)\hat y(1-\hat y).
$$

### Output-layer gradients

Since $z_3=v_1h_1+v_2h_2+c_3$,

$$
\frac{\partial L}{\partial v_1}=\delta_3h_1,
$$

$$
\frac{\partial L}{\partial v_2}=\delta_3h_2,
$$

$$
\frac{\partial L}{\partial c_3}=\delta_3.
$$

### First hidden neuron

The output error reaches the first hidden neuron through the weight $v_1$:

$$
\delta_1
=\frac{\partial L}{\partial z_1}
=\delta_3v_1h_1(1-h_1).
$$

Therefore,

$$
\frac{\partial L}{\partial w_{11}}=\delta_1x_1,
$$

$$
\frac{\partial L}{\partial w_{12}}=\delta_1x_2,
$$

$$
\frac{\partial L}{\partial c_1}=\delta_1.
$$

### Second hidden neuron

Similarly,

$$
\delta_2
=\frac{\partial L}{\partial z_2}
=\delta_3v_2h_2(1-h_2),
$$

and hence

$$
\frac{\partial L}{\partial w_{21}}=\delta_2x_1,
$$

$$
\frac{\partial L}{\partial w_{22}}=\delta_2x_2,
$$

$$
\frac{\partial L}{\partial c_2}=\delta_2.
$$

Backpropagation is therefore an efficient, organized application of the chain rule. It starts from the final prediction error and propagates its influence backward through the output layer and then through the hidden layer.

---

## 14. Why do the factors $h(1-h)$ and $\hat y(1-\hat y)$ appear?

They come directly from the derivative of the sigmoid:

$$
\sigma'(a)=\sigma(a)(1-\sigma(a)).
$$

If a hidden neuron outputs $h$, the derivative of its activation is $h(1-h)$. Likewise, because the output neuron returns $\hat y$, its activation derivative is $\hat y(1-\hat y)$.

These factors measure how sensitive a neuron's output is to a small change in its input. A sigmoid is most sensitive near $0.5$ and becomes less sensitive as its output approaches $0$ or $1$.

---

## 15. Example of a complete parameter update

Combining the derivatives above, the update of $w_{11}$ is

$$
w_{11}\leftarrow w_{11}
-\eta(\hat y-y)\hat y(1-\hat y)
v_1h_1(1-h_1)x_1.
$$

This single expression contains the entire chain of dependencies:

$$
\text{prediction error}
\longrightarrow
\text{output sigmoid}
\longrightarrow
v_1
\longrightarrow
\text{hidden sigmoid}
\longrightarrow
x_1.
$$

The influence of the error is followed backward through the same sequence of operations used during the forward pass. This is precisely why the method is called **backpropagation**.

---

## 16. Training and inference

Training and inference use the same forward computation, but only training modifies the parameters.

### Training

For each example, the network:

1. receives the input $(x_1,x_2)$;
2. computes the hidden values $(h_1,h_2)$;
3. computes the prediction $\hat y$;
4. compares $\hat y$ with the target $y$;
5. evaluates the loss;
6. computes the gradients by backpropagation;
7. updates the weights and biases.

These steps are repeated many times until the loss becomes sufficiently small.

### Inference

After training, the parameters remain fixed. For a new input, the network performs only the forward pass

$$
(x_1,x_2)\longrightarrow(h_1,h_2)\longrightarrow\hat y.
$$

No target value, loss calculation, backpropagation, or parameter update is required.

---

## 17. The parameters are shared by all examples

A common misconception is that every training point has its own weights. In fact, the network has one global parameter vector $\theta$, shared by the entire dataset.

With stochastic gradient descent, for example, the first training example may produce the update

$$
\theta^{(1)}=\theta^{(0)}-\eta\nabla L_1,
$$

and the next example then produces

$$
\theta^{(2)}=\theta^{(1)}-\eta\nabla L_2.
$$

Every example therefore modifies the same network. Over time, the parameters must find a configuration that works well for the dataset as a whole.

---

## 18. What the network learns in this example

In our interpretable solution, the two hidden neurons behave approximately as

$$
h_1\approx
\begin{cases}
0, & x_1+x_2<0.5,\\
1, & x_1+x_2>0.5,
\end{cases}
$$

and

$$
h_2\approx
\begin{cases}
0, & x_1+x_2<1.5,\\
1, & x_1+x_2>1.5.
\end{cases}
$$

Consequently,

- below the first threshold, $(h_1,h_2)\approx(0,0)$;
- between the thresholds, $(h_1,h_2)\approx(1,0)$;
- above both thresholds, $(h_1,h_2)\approx(1,1)$.

For XOR, the desired outputs in these three regions are respectively

$$
0,\qquad1,\qquad0.
$$

The output neuron learns to return a high value in the middle region and a low value in the two outer regions.

---

## 19. The central idea

The two hidden lines are not themselves the output of the network. Their purpose is to create the new quantities $h_1$ and $h_2$.

The sigmoid converts the position of an input relative to each threshold into a soft, almost binary value:

$$
\boxed{
\text{threshold lines in the }(x_1,x_2)\text{ plane}
\longrightarrow
(h_1,h_2)
}.
$$

The output neuron then separates the classes in this new space:

$$
\boxed{
(h_1,h_2)
\longrightarrow
\text{linear boundary in the hidden space}
\longrightarrow
\hat y
}.
$$

It is the nonlinear transformation

$$
(x_1,x_2)\longmapsto(h_1,h_2)
$$

that makes XOR linearly separable.

---

## 20. Complete mathematical summary

### Forward propagation

$$
z_1=w_{11}x_1+w_{12}x_2+c_1,
\qquad
h_1=\sigma(z_1),
$$

$$
z_2=w_{21}x_1+w_{22}x_2+c_2,
\qquad
h_2=\sigma(z_2),
$$

$$
z_3=v_1h_1+v_2h_2+c_3,
\qquad
\hat y=\sigma(z_3).
$$

### Loss

For one example,

$$
L=\frac12(\hat y-y)^2.
$$

For the complete dataset,

$$
L(\theta)=\frac1N\sum_{i=1}^{N}(\hat y_i-y_i)^2.
$$

### Optimization problem

$$
\theta^\star=\arg\min_\theta L(\theta).
$$

### Backpropagation

Compute

$$
\frac{\partial L}{\partial p}
$$

for every parameter $p$ by applying the chain rule from the output layer back to the hidden layer.

### Gradient-descent update

$$
p\leftarrow p-\eta\frac{\partial L}{\partial p}.
$$

Repeating this process allows the network to discover weights and biases that transform the original data into a representation in which classification becomes simple.

---

## 21. The whole idea in one sentence

> A neural network solves XOR by using nonlinear hidden neurons to transform each input from $(x_1,x_2)$ into new coordinates $(h_1,h_2)$, where a single output neuron can linearly separate the two classes.

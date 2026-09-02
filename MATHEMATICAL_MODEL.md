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

The sigmoid transforms any real number into a value between $0$ and $1$:

$$
\sigma(a)\in(0,1).
$$

For a hidden neuron, the quantity

$$
z=w_1x_1+w_2x_2+c
$$

determines the output

$$
h=\sigma(z).
$$

The equation

$$
z=0
$$

is the neuron's threshold line. On this line, $h=0.5$. On one side of the line, $z<0$ and therefore $h<0.5$; on the other side, $z>0$ and therefore $h>0.5$.

Thus, $h$ is a continuous measure of the side of the line on which the point lies. Far into one half-plane, $h$ is close to $0$; far into the other, it is close to $1$; near the line, it takes an intermediate value. For this reason, the sigmoid is called a **soft threshold**.

---

## 5. From lines to new coordinates

The hidden layer contains two neurons, so it associates two new numbers with every point $(x_1,x_2)$:

$$
h_1=\sigma(w_{11}x_1+w_{12}x_2+c_1),
$$

$$
h_2=\sigma(w_{21}x_1+w_{22}x_2+c_2).
$$

These equations define a transformation

$$
T:\mathbb{ℝ}^2\longrightarrow(0,1)^2,
$$

$$
T(x_1,x_2)=(h_1,h_2).
$$

In matrix form,

$$
\begin{pmatrix}
h_1\\
h_2
\end{pmatrix}=
\sigma\left(
\begin{pmatrix}
w_{11} & w_{12}\\
w_{21} & w_{22}
\end{pmatrix}
\begin{pmatrix}
x_1\\
x_2
\end{pmatrix}
+
\begin{pmatrix}
c_1\\
c_2
\end{pmatrix}
\right),
$$

where the sigmoid is applied separately to the two components.

This formula describes a genuine change of representation:

- in the input plane, a point is identified by its coordinates $(x_1,x_2)$;
- in the hidden plane, the same point is represented by the coordinates $(h_1,h_2)$;
- $h_1$ describes the point's position relative to the first threshold line;
- $h_2$ describes its position relative to the second threshold line.

The axes of the hidden plane therefore no longer represent the original variables $x_1$ and $x_2$. They represent the responses of the two hidden neurons.

---

## 6. How the input plane is transformed

The first threshold line divides the input plane according to the value of $h_1$, while the second divides it according to the value of $h_2$. Together, the two lines partition the input plane into regions.

Each region is sent to a different part of the hidden plane:

- points on the negative side of both lines have $h_1\approx0$ and $h_2\approx0$, so they are mapped near $(0,0)$;
- points on the positive side of the first line but the negative side of the second have $h_1\approx1$ and $h_2\approx0$, so they are mapped near $(1,0)$;
- points on the positive side of both lines have $h_1\approx1$ and $h_2\approx1$, so they are mapped near $(1,1)$.

The transformation can therefore be summarized as

$$
\begin{array}{c}
\text{region of the }(x_1,x_2)\text{ plane}
\\[2mm]
\downarrow\;T
\\[2mm]
\text{position in the }(h_1,h_2)\text{ plane}.
\end{array}
$$

The transformation does not physically move a point inside the original plane. Instead, it assigns the point a new pair of coordinates. Two points that are far apart in the input plane can acquire similar hidden coordinates, while points that were difficult to separate in the input plane can become separated in the hidden plane.

<img src="xor_input_space.png" width="500" alt="XOR points and hidden-neuron threshold lines">

---

## 7. The transformation used for XOR

For XOR, the two threshold lines can be placed at

$$
x_1+x_2=0.5
$$

and

$$
x_1+x_2=1.5.
$$

The first hidden coordinate indicates whether a point has crossed the first line, and the second indicates whether it has crossed the second. Consequently, the three regions are transformed as follows:

| Region in the $(x_1,x_2)$ plane | Hidden coordinates | XOR class |
|:---|:---:|:---:|
| $x_1+x_2<0.5$ | $(h_1,h_2)\approx(0,0)$ | $0$ |
| $0.5<x_1+x_2<1.5$ | $(h_1,h_2)\approx(1,0)$ | $1$ |
| $x_1+x_2>1.5$ | $(h_1,h_2)\approx(1,1)$ | $0$ |

The two regions belonging to class $0$ are separated in the original plane by the region belonging to class $1$. After the transformation, however, the entire class-$1$ region is represented near $(1,0)$, away from the class-$0$ regions.

The network may learn different threshold lines, but their purpose remains the same: to construct hidden coordinates that make the two classes easier to separate.

---

## 8. Interpreting $h_1$ and $h_2$

The hidden values are not predictions. They are the new coordinates produced by the transformation $T$.

The first coordinate $h_1$ contains information about the first threshold line; the second coordinate $h_2$ contains information about the second. The pair $(h_1,h_2)$ therefore summarizes the position of the original point relative to both learned boundaries.

This is the essential difference between the two planes:

$$
\underbrace{(x_1,x_2)}_{\text{original features}}
\quad\xrightarrow{\;T\;}\quad
\underbrace{(h_1,h_2)}_{\text{features learned by the network}}.
$$

---

## 9. Linear separation in the hidden space

<img src="xor_hidden_space.png" width="500" alt="XOR points in the hidden space">

In the hidden plane, the class-$1$ points are concentrated near $(1,0)$, while the class-$0$ points lie near $(0,0)$ and $(1,1)$. The two classes, which were not linearly separable in the input plane, are now linearly separable.

The line

$$
h_1-h_2-0.5=0
$$

separates the class-$1$ region from the class-$0$ regions. The output neuron can therefore compute

$$
\hat y=\sigma\left(K(h_1-h_2-0.5)\right),
$$

where $K>0$ is sufficiently large to make the final sigmoid behave almost like a binary classifier.

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

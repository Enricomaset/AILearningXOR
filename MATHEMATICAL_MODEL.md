# XOR from Scratch: The Mathematics of a 2 → 2 → 1 Neural Network

This page develops, step by step, a small neural network capable of learning the **XOR** function.

The goal is not to rely on a machine-learning library, but to understand what the network is doing geometrically and mathematically. In particular, we will see how neurons, weights, biases, activation functions, loss minimization, gradient descent, and backpropagation work together.

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

The purpose of the hidden layer is to transform the original points into a new space in which the two classes **can** be separated by a single line.

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

The network has nine trainable parameters:

$$
w_{11},\;w_{12},\;w_{21},\;w_{22},\;c_1,\;c_2,\;v_1,\;v_2,\;c_3.
$$

---

## 3. The geometric meaning of a hidden neuron

Before applying the activation function, the first hidden neuron computes

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

defines a straight line in the $(x_1,x_2)$ plane.

Similarly, the second hidden neuron defines the line

$$
w_{21}x_1+w_{22}x_2+c_2=0.
$$

<img src="xor_input_space.png" width="500" alt="XOR points and hidden-neuron threshold lines">

The weights determine the orientation of each line, while the bias allows the line to move away from the origin.

It is important to distinguish the lines from the neuron outputs. The lines are the sets of points for which the inputs of the sigmoids are zero. The quantities $h_1$ and $h_2$ are instead the values obtained **after** the activation function is applied.

---

## 4. The sigmoid activation function

The activation function used here is the sigmoid

$$
\sigma(a)=\frac{1}{1+e^{-a}}.
$$

It maps every real number into the interval

$$
\sigma(a)\in(0,1).
$$

For a hidden neuron,

$$
z=w_1x_1+w_2x_2+c,
$$

and

$$
h=\sigma(z).
$$

The equation

$$
z=0
$$

is the neuron's threshold line.

On this line,

$$
h=0.5.
$$

On one side,

$$
z<0
\quad\Longrightarrow\quad
h<0.5,
$$

while on the other,

$$
z>0
\quad\Longrightarrow\quad
h>0.5.
$$

Far from the line, the sigmoid behaves almost like a binary indicator:

$$
h\approx
\begin{cases}
0, & \text{on one side},\\
1, & \text{on the other side}.
\end{cases}
$$

For this reason, the sigmoid can be interpreted as a **soft threshold**.

---

## 5. Transforming the input space

The two hidden neurons associate two new values with every input point:

$$
h_1=\sigma(w_{11}x_1+w_{12}x_2+c_1),
$$

$$
h_2=\sigma(w_{21}x_1+w_{22}x_2+c_2).
$$

Thus, the hidden layer transforms

$$
(x_1,x_2)
$$

into

$$
(h_1,h_2).
$$

In matrix form,

$$
\begin{pmatrix}
h_1\\
h_2
\end{pmatrix}
=
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

where the sigmoid is applied component-wise.

Each hidden neuron tells us approximately on which side of its threshold line the point lies.

Therefore, the pair

$$
(h_1,h_2)
$$

can be interpreted as a new pair of coordinates describing the position of the original point relative to the two learned lines.

The important idea is that the network does not move the point inside the original plane. It gives the point a **new representation**.

So the hidden layer performs

$$
\boxed{
(x_1,x_2)
\longrightarrow
(h_1,h_2)
}.
$$

---

## 6. Why the transformation is useful

In the original $(x_1,x_2)$ plane, XOR cannot be separated by a single straight line:

$$
\boxed{
(x_1,x_2)\text{ space}
\quad\longrightarrow\quad
\text{not linearly separable}.
}
$$

The hidden layer changes the representation of the four points.

After the transformation, the same inputs become four points in the $(h_1,h_2)$ plane.

Training adjusts the hidden weights and biases so that the transformed points can be divided into the two classes by one straight line.

Thus,

$$
\boxed{
(h_1,h_2)\text{ space}
\quad\longrightarrow\quad
\text{linearly separable}.
}
$$

The central geometric idea is

$$
\boxed{
\begin{array}{c}
\text{original space}\\
\text{one line is not enough}
\end{array}
}
\quad
\longrightarrow
\quad
\boxed{
\begin{array}{c}
\text{hidden transformation}\\
(x_1,x_2)\mapsto(h_1,h_2)
\end{array}
}
\quad
\longrightarrow
\quad
\boxed{
\begin{array}{c}
\text{hidden space}\\
\text{one line is enough}
\end{array}
}.
$$

---

## 7. An interpretable XOR transformation

One possible solution places the two hidden threshold lines at

$$
x_1+x_2=0.5
$$

and

$$
x_1+x_2=1.5.
$$

Then the hidden neurons behave approximately as

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

Therefore,

| Region in the input plane | Hidden coordinates | XOR class |
|:---|:---:|:---:|
| $x_1+x_2<0.5$ | $(0,0)$ approximately | $0$ |
| $0.5<x_1+x_2<1.5$ | $(1,0)$ approximately | $1$ |
| $x_1+x_2>1.5$ | $(1,1)$ approximately | $0$ |

The hidden layer has therefore changed the geometry of the data.

---

## 8. Separation in the hidden space

<img src="xor_hidden_space.png" width="500" alt="XOR points in the hidden space">

Once the points have been transformed into the $(h_1,h_2)$ plane, the output neuron only has to divide this plane into two regions.

It does this using the line

$$
v_1h_1+v_2h_2+c_3=0.
$$

If

$$
v_1h_1+v_2h_2+c_3>0,
$$

then

$$
\hat y\approx1.
$$

If

$$
v_1h_1+v_2h_2+c_3<0,
$$

then

$$
\hat y\approx0.
$$

The output neuron therefore decides the class according to which side of the line the transformed point lies.

For example, the line

$$
h_1-h_2-0.5=0
$$

can separate the XOR classes in the hidden plane.

The output neuron may then compute

$$
\hat y=
\sigma\left(K(h_1-h_2-0.5)\right),
$$

where $K>0$ is sufficiently large.

The complete geometric mechanism is therefore

$$
\boxed{
(x_1,x_2)
\longrightarrow
(h_1,h_2)
\longrightarrow
\text{one separating line}
\longrightarrow
\hat y
}.
$$

The hidden layer changes the representation.

The output layer checks on which side of a line the transformed point lies.

---

## 9. The output neuron

In general, the output neuron computes

$$
\hat y=\sigma(v_1h_1+v_2h_2+c_3).
$$

It does not receive $(x_1,x_2)$ directly.

It receives only

$$
(h_1,h_2).
$$

The parameters therefore have different roles.

The parameters

$$
w_{11},w_{12},w_{21},w_{22},c_1,c_2
$$

determine how the original input plane is transformed.

The parameters

$$
v_1,v_2,c_3
$$

determine the separating line in the hidden plane.

All these parameters are learned together during training.

---

## 10. The loss function

To train the network, we need a numerical measure of the difference between the prediction $\hat y$ and the correct value $y$.

A simple choice is the **mean squared error**:

$$
L=
\frac{1}{N}
\sum_{i=1}^{N}
(\hat y_i-y_i)^2.
$$

For XOR,

$$
\begin{aligned}
L=\frac14\big[&
\hat y(0,0)^2
+(\hat y(0,1)-1)^2\\
&+(\hat y(1,0)-1)^2
+\hat y(1,1)^2
\big].
\end{aligned}
$$

Training consists of finding values of

$$
w_{11},w_{12},w_{21},w_{22},
c_1,c_2,v_1,v_2,c_3
$$

that make the loss as small as possible.

The network is not explicitly told where to place the threshold lines.

It receives only the input examples and the correct outputs.

The positions and orientations of the lines emerge from the adjustment of the weights and biases.

---

## 11. Gradient descent

Gradient descent changes every parameter in the direction that reduces the loss.

For example,

$$
w_{11}
\leftarrow
w_{11}
-\eta
\frac{\partial L}{\partial w_{11}},
$$

$$
w_{12}
\leftarrow
w_{12}
-\eta
\frac{\partial L}{\partial w_{12}},
$$

$$
w_{21}
\leftarrow
w_{21}
-\eta
\frac{\partial L}{\partial w_{21}},
$$

$$
w_{22}
\leftarrow
w_{22}
-\eta
\frac{\partial L}{\partial w_{22}},
$$

$$
c_1
\leftarrow
c_1
-\eta
\frac{\partial L}{\partial c_1},
$$

$$
c_2
\leftarrow
c_2
-\eta
\frac{\partial L}{\partial c_2},
$$

$$
v_1
\leftarrow
v_1
-\eta
\frac{\partial L}{\partial v_1},
$$

$$
v_2
\leftarrow
v_2
-\eta
\frac{\partial L}{\partial v_2},
$$

$$
c_3
\leftarrow
c_3
-\eta
\frac{\partial L}{\partial c_3}.
$$

The learning rate $\eta>0$ determines the size of each update.

Updates to

$$
w_{11},w_{12},w_{21},w_{22},c_1,c_2
$$

change the transformation of the input space.

Updates to

$$
v_1,v_2,c_3
$$

change the separating line in the hidden space.

---

## 12. Backpropagation

For one training example, define

$$
L=\frac12(\hat y-y)^2.
$$

The network computes

$$
z_1=w_{11}x_1+w_{12}x_2+c_1,
$$

$$
h_1=\sigma(z_1),
$$

$$
z_2=w_{21}x_1+w_{22}x_2+c_2,
$$

$$
h_2=\sigma(z_2),
$$

$$
z_3=v_1h_1+v_2h_2+c_3,
$$

$$
\hat y=\sigma(z_3).
$$

The derivative of the sigmoid is

$$
\sigma'(a)=\sigma(a)(1-\sigma(a)).
$$

Since

$$
L=\frac12(\hat y-y)^2,
$$

we first obtain

$$
\frac{\partial L}{\partial \hat y}
=
\hat y-y.
$$

Also,

$$
\frac{\partial \hat y}{\partial z_3}
=
\hat y(1-\hat y).
$$

Therefore,

$$
\frac{\partial L}{\partial z_3}
=
(\hat y-y)\hat y(1-\hat y).
$$

### Output-layer gradients

Since

$$
z_3=v_1h_1+v_2h_2+c_3,
$$

we have

$$
\frac{\partial z_3}{\partial v_1}=h_1,
$$

$$
\frac{\partial z_3}{\partial v_2}=h_2,
$$

$$
\frac{\partial z_3}{\partial c_3}=1.
$$

Hence,

$$
\boxed{
\frac{\partial L}{\partial v_1}
=
(\hat y-y)\hat y(1-\hat y)h_1
}
$$

$$
\boxed{
\frac{\partial L}{\partial v_2}
=
(\hat y-y)\hat y(1-\hat y)h_2
}
$$

$$
\boxed{
\frac{\partial L}{\partial c_3}
=
(\hat y-y)\hat y(1-\hat y)
}
$$

### First hidden neuron

Because

$$
z_3=v_1h_1+v_2h_2+c_3,
$$

we have

$$
\frac{\partial z_3}{\partial h_1}=v_1.
$$

Since

$$
h_1=\sigma(z_1),
$$

we have

$$
\frac{\partial h_1}{\partial z_1}
=
h_1(1-h_1).
$$

Also,

$$
\frac{\partial z_1}{\partial w_{11}}=x_1,
$$

$$
\frac{\partial z_1}{\partial w_{12}}=x_2,
$$

$$
\frac{\partial z_1}{\partial c_1}=1.
$$

Therefore,

$$
\boxed{
\frac{\partial L}{\partial w_{11}}
=
(\hat y-y)
\hat y(1-\hat y)
v_1
h_1(1-h_1)
x_1
}
$$

$$
\boxed{
\frac{\partial L}{\partial w_{12}}
=
(\hat y-y)
\hat y(1-\hat y)
v_1
h_1(1-h_1)
x_2
}
$$

$$
\boxed{
\frac{\partial L}{\partial c_1}
=
(\hat y-y)
\hat y(1-\hat y)
v_1
h_1(1-h_1)
}
$$

### Second hidden neuron

Similarly,

$$
\frac{\partial z_3}{\partial h_2}=v_2,
$$

and

$$
\frac{\partial h_2}{\partial z_2}
=
h_2(1-h_2).
$$

Since

$$
\frac{\partial z_2}{\partial w_{21}}=x_1,
$$

$$
\frac{\partial z_2}{\partial w_{22}}=x_2,
$$

$$
\frac{\partial z_2}{\partial c_2}=1,
$$

we obtain

$$
\boxed{
\frac{\partial L}{\partial w_{21}}
=
(\hat y-y)
\hat y(1-\hat y)
v_2
h_2(1-h_2)
x_1
}
$$

$$
\boxed{
\frac{\partial L}{\partial w_{22}}
=
(\hat y-y)
\hat y(1-\hat y)
v_2
h_2(1-h_2)
x_2
}
$$

$$
\boxed{
\frac{\partial L}{\partial c_2}
=
(\hat y-y)
\hat y(1-\hat y)
v_2
h_2(1-h_2)
}
$$

Backpropagation is therefore simply the repeated application of the chain rule from the output of the network back toward the input.

---

## 13. Why do the factors $h(1-h)$ and $\hat y(1-\hat y)$ appear?

They come directly from the derivative of the sigmoid:

$$
\sigma'(a)
=
\sigma(a)(1-\sigma(a)).
$$

Since

$$
h_1=\sigma(z_1),
$$

we have

$$
\frac{\partial h_1}{\partial z_1}
=
h_1(1-h_1).
$$

Similarly,

$$
\frac{\partial h_2}{\partial z_2}
=
h_2(1-h_2),
$$

and

$$
\frac{\partial \hat y}{\partial z_3}
=
\hat y(1-\hat y).
$$

These factors measure how strongly the output of a sigmoid changes when its input changes.

---

## 14. Complete parameter updates

Substituting the derivatives directly into gradient descent gives all the updates explicitly.

For the output layer,

$$
\boxed{
v_1
\leftarrow
v_1
-
\eta
(\hat y-y)
\hat y(1-\hat y)
h_1
}
$$

$$
\boxed{
v_2
\leftarrow
v_2
-
\eta
(\hat y-y)
\hat y(1-\hat y)
h_2
}
$$

$$
\boxed{
c_3
\leftarrow
c_3
-
\eta
(\hat y-y)
\hat y(1-\hat y)
}
$$

For the first hidden neuron,

$$
\boxed{
w_{11}
\leftarrow
w_{11}
-
\eta
(\hat y-y)
\hat y(1-\hat y)
v_1
h_1(1-h_1)
x_1
}
$$

$$
\boxed{
w_{12}
\leftarrow
w_{12}
-
\eta
(\hat y-y)
\hat y(1-\hat y)
v_1
h_1(1-h_1)
x_2
}
$$

$$
\boxed{
c_1
\leftarrow
c_1
-
\eta
(\hat y-y)
\hat y(1-\hat y)
v_1
h_1(1-h_1)
}
$$

For the second hidden neuron,

$$
\boxed{
w_{21}
\leftarrow
w_{21}
-
\eta
(\hat y-y)
\hat y(1-\hat y)
v_2
h_2(1-h_2)
x_1
}
$$

$$
\boxed{
w_{22}
\leftarrow
w_{22}
-
\eta
(\hat y-y)
\hat y(1-\hat y)
v_2
h_2(1-h_2)
x_2
}
$$

$$
\boxed{
c_2
\leftarrow
c_2
-
\eta
(\hat y-y)
\hat y(1-\hat y)
v_2
h_2(1-h_2)
}
$$

These formulas show explicitly how the prediction error changes every weight and bias in the network.

---

## 15. Training and inference

Training and inference use the same forward computation, but only training modifies the parameters.

### Training

For each example, the network:

1. receives $(x_1,x_2)$;
2. computes $h_1$ and $h_2$;
3. computes $\hat y$;
4. compares $\hat y$ with $y$;
5. computes the loss;
6. computes all derivatives;
7. updates all weights and biases.

These operations are repeated many times.

### Inference

After training, the weights and biases remain fixed.

For a new input, the network only computes

$$
(x_1,x_2)
\longrightarrow
(h_1,h_2)
\longrightarrow
\hat y.
$$

No loss, derivatives, or parameter updates are required.

---

## 16. The same parameters are used for every example

The four XOR examples do not have separate weights.

The same values

$$
w_{11},w_{12},w_{21},w_{22},
c_1,c_2,v_1,v_2,c_3
$$

are used for all four points.

For example, after processing one point, the network may update

$$
w_{11}
\leftarrow
w_{11}
-
\eta
\frac{\partial L}{\partial w_{11}}.
$$

The next training point then uses this new value of $w_{11}$.

Therefore, all training examples progressively modify the same network.

The final weights and biases must produce good predictions for the dataset as a whole.

---

## 17. What the network learns

For XOR, the hidden layer learns a transformation

$$
(x_1,x_2)
\longrightarrow
(h_1,h_2)
$$

that reorganizes the four points.

In the original plane, the two classes cannot be separated by one line.

After the transformation, they can.

The final neuron therefore has a much simpler task:

$$
\boxed{
\text{determine on which side of a line the transformed point lies}.
}
$$

This is the key reason why adding a nonlinear hidden layer allows the network to solve XOR.

---

## 18. The central idea

The hidden neurons do not directly produce the final classification.

They create new coordinates:

$$
\boxed{
(x_1,x_2)
\longrightarrow
(h_1,h_2)
}.
$$

These coordinates reorganize the geometry of the data.

The output neuron then uses one line

$$
\boxed{
v_1h_1+v_2h_2+c_3=0
}
$$

to divide the hidden plane into two regions.

A transformed point on one side of the line corresponds to one class, while a transformed point on the other side corresponds to the other.

Thus,

$$
\boxed{
\begin{array}{c}
\text{XOR in the input plane}\\
\text{not separable by one line}
\end{array}
}
\quad
\longrightarrow
\quad
\boxed{
\begin{array}{c}
\text{nonlinear transformation}\\
(x_1,x_2)\mapsto(h_1,h_2)
\end{array}
}
\quad
\longrightarrow
\quad
\boxed{
\begin{array}{c}
\text{hidden plane}\\
\text{separable by one line}
\end{array}
}.
$$

---

## 19. Complete mathematical summary

### Forward propagation

$$
h_1
=
\sigma(w_{11}x_1+w_{12}x_2+c_1),
$$

$$
h_2
=
\sigma(w_{21}x_1+w_{22}x_2+c_2),
$$

$$
\hat y
=
\sigma(v_1h_1+v_2h_2+c_3).
$$

### Loss

For one example,

$$
L=\frac12(\hat y-y)^2.
$$

For the complete dataset,

$$
L=
\frac1N
\sum_{i=1}^{N}
(\hat y_i-y_i)^2.
$$

### Output-layer derivatives

$$
\frac{\partial L}{\partial v_1}
=
(\hat y-y)\hat y(1-\hat y)h_1,
$$

$$
\frac{\partial L}{\partial v_2}
=
(\hat y-y)\hat y(1-\hat y)h_2,
$$

$$
\frac{\partial L}{\partial c_3}
=
(\hat y-y)\hat y(1-\hat y).
$$

### Hidden-layer derivatives

$$
\frac{\partial L}{\partial w_{11}}
=
(\hat y-y)\hat y(1-\hat y)
v_1h_1(1-h_1)x_1,
$$

$$
\frac{\partial L}{\partial w_{12}}
=
(\hat y-y)\hat y(1-\hat y)
v_1h_1(1-h_1)x_2,
$$

$$
\frac{\partial L}{\partial c_1}
=
(\hat y-y)\hat y(1-\hat y)
v_1h_1(1-h_1),
$$

$$
\frac{\partial L}{\partial w_{21}}
=
(\hat y-y)\hat y(1-\hat y)
v_2h_2(1-h_2)x_1,
$$

$$
\frac{\partial L}{\partial w_{22}}
=
(\hat y-y)\hat y(1-\hat y)
v_2h_2(1-h_2)x_2,
$$

$$
\frac{\partial L}{\partial c_2}
=
(\hat y-y)\hat y(1-\hat y)
v_2h_2(1-h_2).
$$

### Gradient-descent updates

Every parameter is updated by subtracting its derivative multiplied by the learning rate $\eta$.

For example,

$$
w_{11}
\leftarrow
w_{11}
-
\eta
(\hat y-y)
\hat y(1-\hat y)
v_1h_1(1-h_1)x_1.
$$

Repeating these updates changes both the transformation produced by the hidden layer and the separating line produced by the output layer.

---

## 20. The whole idea in one sentence

> A neural network solves XOR by transforming the original coordinates $(x_1,x_2)$ into new coordinates $(h_1,h_2)$ in which the two classes lie on opposite sides of a single straight line.

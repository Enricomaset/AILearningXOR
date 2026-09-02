# XOR from Scratch: The Mathematics of a 2 → 2 → 1 Neural Network

This page develops, step by step, a small neural network capable of
learning **XOR**.

The goal is not to rely on a machine-learning library, but to understand
geometrically and mathematically what the following components actually
do:

- neurons, weights and biases;
- the activation function (sigmoid);
- the transformation from the $(x_1,x_2)$ plane to the $(h_1,h_2)$ plane;
- the loss function;
- gradient descent;
- backpropagation.

---
## 1. The XOR problem

We want to learn the XOR function:
| $x_1$ | $x_2$ | $y$ |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

In the $(x_1,x_2)$ plane, the two points with output $1$ lie on opposite
corners of the square, while the two points with output $0$ lie on the
other two corners.

A single straight line cannot separate the two points with $y=1$ from
the two points with $y=0$.

Therefore, a single linear neuron is not sufficient.

---
## 2. Network architecture

We use a

$$
2 \longrightarrow 2 \longrightarrow 1
$$

neural network consisting of:

- two inputs: $(x_1,x_2)$;
- two hidden neurons: $(h_1,h_2)$;
- one output neuron: $\hat y$.

The hidden neurons are

$$
h_1=\sigma(w_{11}x_1+w_{12}x_2+c_1),
$$

$$
h_2=\sigma(w_{21}x_1+w_{22}x_2+c_2),
$$

and the output neuron is

$$
\hat y=\sigma(v_1h_1+v_2h_2+c_3).
$$

The trainable parameters are therefore

$$
\theta=
(w_{11},w_{12},w_{21},w_{22},c_1,c_2,v_1,v_2,c_3).
$$

There are 9 trainable parameters in total.

---
## 3. What does a hidden neuron represent geometrically?

Before applying the sigmoid, the first hidden neuron computes

$$
w_{11}x_1+w_{12}x_2+c_1.
$$

Setting this quantity equal to zero gives

$$
w_{11}x_1+w_{12}x_2+c_1=0,
$$

which is a **straight line** in the

$$
(x_1,x_2)
$$

plane.

Similarly, the second neuron defines

$$
w_{21}x_1+w_{22}x_2+c_2=0.
$$

We therefore have two lines.

<img src="xor_input_space.png" width="500">

The weights mainly determine the orientation of each line, while the
bias allows the line to be translated. Without a bias, the line would be
forced to pass through the origin.

An important distinction is that $h_1$ and $h_2$ are **not themselves
the lines**. The lines are the sets of points for which the arguments of
the corresponding sigmoids are zero. The quantities $h_1$ and $h_2$ are
the outputs produced after applying the activation function.

---
## 4. The sigmoid activation function

We use the sigmoid

$$
\sigma(a)=\frac{1}{1+e^{-a}}.
$$

Its behavior is

$$
a\ll0 \quad\Rightarrow\quad
\sigma(a)\approx0,
$$

$$
a=0 \quad\Rightarrow\quad
\sigma(a)=0.5,
$$

$$
a\gg0 \quad\Rightarrow\quad
\sigma(a)\approx1.
$$

Consider

$$
h=\sigma(w_1x_1+w_2x_2+c).
$$

The line

$$
w_1x_1+w_2x_2+c=0
$$

divides the plane into two half-planes.

If

$$
w_1x_1+w_2x_2+c<0,
$$

the point lies on one side of the line and (h) tends toward (0).

If

$$
w_1x_1+w_2x_2+c>0,
$$

the point lies on the other side and (h) tends toward (1).

Thus, the sigmoid acts as a **soft threshold**. It does not usually
return exactly 0 or 1, but it can return values extremely close to them.

---
## 6. An intuitive geometric solution to XOR

To understand what the network can learn, consider the following two
hidden neurons:

$$
h_1=\sigma(10x_1+10x_2-5),
$$

$$
h_2=\sigma(10x_1+10x_2-15).
$$

Their threshold lines are

$$
10x_1+10x_2-5=0,
$$

or

$$
x_1+x_2=0.5,
$$

and

$$
10x_1+10x_2-15=0,
$$

or

$$
x_1+x_2=1.5.
$$

These two parallel lines divide the XOR points into three regions.

The training process is free to find other equivalent parameter
configurations; these particular lines are simply a convenient and
geometrically transparent solution.

---
## 7. Computing $h_1$ and $h_2$

### Input $(0,0)$

For the first neuron,

$$
h_1=\sigma(-5)\approx0.0067.
$$

For the second,

$$
h_2=\sigma(-15)\approx0.
$$

Therefore,

$$
(0,0)\longrightarrow$(h_1,h_2)$\approx(0,0).
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
(0,1)\longrightarrow$(h_1,h_2)$\approx(1,0).
$$

### Input $(1,0)$

By symmetry,

$$
(1,0)\longrightarrow$(h_1,h_2)$\approx(1,0).
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
(1,1)\longrightarrow$(h_1,h_2)$\approx(1,1).
$$

The complete transformation is approximately:

| Input $(x_1,x_2)$ | $h_1$ | $h_2$ | Target $y$ |
|:---:|:---:|:---:|:---:|
| $(0,0)$ | 0.0067 | ~0 | 0 |
| $(0,1)$ | 0.9933 | 0.0067 | 1 |
| $(1,0)$ | 0.9933 | 0.0067 | 1 |
| $(1,1)$ | ~1 | 0.9933 | 0 |

------------------- -------- -------- ------------
  $(0,0)$               0.0067      ~0            0
  $(0,1)$               0.9933   0.0067            1
  $(1,0)$               0.9933   0.0067            1
  $(1,1)$                  ~1   0.9933            0

---
## 8. What do $h_1$ and $h_2$ tell us?

The values $h_1$ and $h_2$ are **not yet the final prediction**.

They encode the position of the input point relative to the two
threshold lines.

Approximately,

$$
(h_1,h_2)\approx(0,0)
$$

means that the point is on the "0 side" of both thresholds;

$$
$(h_1,h_2)$\approx(1,0)
$$

means that the point has crossed the first threshold but not the second;

$$
$(h_1,h_2)$\approx(1,1)
$$

means that the point has crossed both thresholds.

The hidden layer therefore converts the original position of the point
into **two new features**.

---
## 9. The transformation into the $(h_1,h_2)$ plane

After the hidden layer, we no longer need to think directly in terms of
the original coordinates

$$
(x_1,x_2)
$$

.

Each input is transformed according to

$$
(x_1,x_2)\longrightarrow(h_1,h_2).
$$

For our example,

$$
(0,0)\to(0,0),
$$

$$
(0,1)\to(1,0),
$$

$$
(1,0)\to(1,0),
$$

$$
(1,1)\to(1,1),
$$

approximately.

<img src="xor_hidden_space.png" width="500">

Now the two positive XOR examples are concentrated near

$$
(h_1,h_2)=(1,0),
$$

while the negative examples lie near

$$
(0,0) \quad\text{and}\quad (1,1).
$$

In this transformed space the classes are linearly separable.

For example, one possible separating line is

$$
h_1-h_2-0.5=0.
$$

Thus the output neuron could behave approximately as

$$
\hat y=
\sigma\left(K(h_1-h_2-0.5)\right),
$$

with a sufficiently large $K>0$.

This is the central geometric idea:

$$
\boxed{
(x_1,x_2)
\;\xrightarrow{\text{hidden layer + nonlinear activation}}\;
(h_1,h_2)
\;\xrightarrow{\text{linear separation + sigmoid}}\;
\hat y
}
$$

The hidden layer **transforms the space**.
The output layer performs the final classification in that transformed
space.

---
## 10. The output neuron

In general,

$$
\hat y=\sigma(v_1h_1+v_2h_2+c_3).
$$

The parameters $(v_1,v_2,c_3)$ are learned during training.

The output neuron does not directly see $(x_1,x_2)$. It receives only $(h_1,h_2)$.

A useful interpretation is therefore:

- $(w_{ij},c_1,c_2)$ learn **how to transform the original input
    space**;
- $(v_1,v_2,c_3)$ learn **how to classify the transformed points**.

---
## 11. The loss function

The network needs a numerical measure of how far its prediction
(\hat y) is from the correct value $y$.

A simple choice is the **Mean Squared Error (MSE)**:

$$
L(\theta) = \frac{1}{N} \sum_{i=1}^{N}
(\hat y_i-y_i)^2.
$$

For XOR there are four training examples:

$$
L(\theta)=\frac14 \left[
\hat y(0,0;\theta)^2
+
(\hat y(0,1;\theta)-1)^2
+
(\hat y(1,0;\theta)-1)^2
+
\hat y(1,1;\theta)^2
\right].
$$

The training problem is

$$
\theta^\star =
\arg\min_\theta L(\theta).
$$

The network is **not explicitly told**:

> Put the first line at (x_1+x_2=0.5) and the second line at
> (x_1+x_2=1.5).

Instead, it receives examples and adjusts its parameters so as to reduce
the loss.

---
## 12. Gradient descent

To minimize the loss, each parameter is updated in the direction
opposite to the gradient:

$$
p \leftarrow
p-\eta\frac{\partial L}{\partial p},
$$

where:

- \(p\) is any weight or bias;
- (\eta) is the learning rate;
- (\frac{\partial L}{\partial p}) tells us how the loss
    changes when $p$ changes.

For all parameters simultaneously,

$$
\theta_{k+1} =
\theta_k-\eta\nabla L(\theta_k).
$$

During training, both the hidden-layer parameters

$$
w_{11},w_{12},w_{21},w_{22},c_1,c_2
$$

and the output-layer parameters

$$
v_1,v_2,c_3
$$

are adjusted.

Geometrically, changing the hidden weights and biases changes the
position and orientation of the hidden threshold lines.

---
## 13. Backpropagation

For clarity, consider the loss for a single training example:

$$
L=\frac12(\hat y-y)^2.
$$

The derivative of the sigmoid is

$$
\sigma'(a) = \sigma(a)(1-\sigma(a)).
$$

Because the output of the final sigmoid is (\hat y), its
derivative contributes

$$
\hat y(1-\hat y).
$$

### Gradients of the output layer

$$
\frac{\partial L}{\partial v_1} =
(\hat y-y)\hat y(1-\hat y)h_1,
$$

$$
\frac{\partial L}{\partial v_2} =
(\hat y-y)\hat y(1-\hat y)h_2,
$$

$$
\frac{\partial L}{\partial c_3} =
(\hat y-y)\hat y(1-\hat y).
$$

### Gradients of the first hidden neuron

$$
\frac{\partial L}{\partial w_{11}} =
(\hat y-y)\hat y(1-\hat y) v_1h_1(1-h_1)x_1,
$$

$$
\frac{\partial L}{\partial w_{12}} =
(\hat y-y)\hat y(1-\hat y) v_1h_1(1-h_1)x_2,
$$

$$
\frac{\partial L}{\partial c_1} =
(\hat y-y)\hat y(1-\hat y) v_1h_1(1-h_1).
$$

### Gradients of the second hidden neuron

$$
\frac{\partial L}{\partial w_{21}} =
(\hat y-y)\hat y(1-\hat y) v_2h_2(1-h_2)x_1,
$$

$$
\frac{\partial L}{\partial w_{22}} =
(\hat y-y)\hat y(1-\hat y) v_2h_2(1-h_2)x_2,
$$

$$
\frac{\partial L}{\partial c_2} =
(\hat y-y)\hat y(1-\hat y) v_2h_2(1-h_2).
$$

Backpropagation is an efficient application of the **chain rule**: it
starts from the final error and propagates its effect backward through
the network.

---
## 14. Why do (h(1-h)) and (\hat y(1-\hat y)) appear?

Because

$$
\sigma'(a)=\sigma(a)(1-\sigma(a)).
$$

If a hidden neuron produces $h_1$, the derivative of its sigmoid is
therefore

$$
h_1(1-h_1).
$$

Similarly, if the output neuron produces (\hat y), its sigmoid
derivative is

$$
\hat y(1-\hat y).
$$

These factors describe how sensitive the output of each neuron is to a
small change in its input.

---
## 15. Example of a complete parameter update

For example,

$$
w_{11} \leftarrow w_{11} - \eta (\hat y-y)
\hat y(1-\hat y) v_1 h_1(1-h_1) x_1.
$$

This single equation contains the entire chain of dependencies:

$$
\text{final error} \rightarrow
\text{output sigmoid} \rightarrow v_1
\rightarrow \text{hidden sigmoid} \rightarrow
x_1.
$$

This is why the procedure is called **backpropagation**: information
about the final error is propagated backward through the network.

---
## 16. Training and inference are different

### Training

During training:

1. choose an input
$$
(x_1,x_2)
$$
;
2. compute $(h_1,h_2)$;
3. compute (\hat y);
4. compare (\hat y) with $y$;
5. compute the loss;
6. compute the gradients;
7. update weights and biases.

This process is repeated many times.

### Inference

After training, the parameters remain fixed.

For a new input, we only compute
$$
$(x_1,x_2)$ \longrightarrow $(h_1,h_2)$ \longrightarrow
\hat y.
$$
No loss minimization and no weight update are performed during
inference.

---
## 17. Weights do not belong to individual data points

A common misconception is that each training point has its own set of
weights.

It does not.

There is one global parameter vector
$$
\theta=(w,c,v,\ldots)
$$
shared by all training examples.

With stochastic gradient descent, schematically,
$$
\theta^{(1)} = \theta^{(0)}
-\eta\nabla L_1,
$$
then
$$
\theta^{(2)} = \theta^{(1)}
-\eta\nabla L_2,
$$
and so on.

Every training example modifies the **same** network parameters.

---
## 18. What is the network actually learning?

In our interpretable example, the hidden layer behaves approximately as
$$
h_1\approx
\begin{cases}
0, & x_1+x_2<0.5,\
1, & x_1+x_2>0.5,
\end{cases}
$$
and
$$
h_2\approx
\begin{cases}
0, & x_1+x_2<1.5,\
1, & x_1+x_2>1.5.
\end{cases}
$$
Therefore:

- below the first threshold: ($(h_1,h_2)$\approx(0,0));
- between the thresholds: ($(h_1,h_2)$\approx(1,0));
- above both thresholds: ($(h_1,h_2)$\approx(1,1)).

For XOR, the desired outputs in these three regions are
$$
0,\qquad1,\qquad0.
$$
The output neuron learns to produce a high value for the middle region
and a low value for the other two.

---
## 19. The central conceptual point

The two hidden lines are **not directly the output of the network**.

Their purpose is to construct two new quantities,
$$
h_1,\qquad h_2.
$$
The sigmoid converts the position relative to each threshold into a
soft, almost binary value.

Thus,
$$
\boxed{
\text{lines in the }$(x_1,x_2)$\text{ plane}
\longrightarrow
$(h_1,h_2)$
}
$$
and then
$$
\boxed{
$(h_1,h_2)$
\longrightarrow
\text{linear boundary in the new space}
\longrightarrow
\hat y
}
$$
The nonlinear transformation
$$
$(x_1,x_2)$\mapsto$(h_1,h_2)$
$$
is what makes XOR linearly separable in the hidden representation.

---
## 20. Complete mathematical summary

### Forward propagation
$$
h_1=\sigma(w_{11}x_1+w_{12}x_2+c_1),
$$
$$
h_2=\sigma(w_{21}x_1+w_{22}x_2+c_2),
$$
$$
\hat y=\sigma(v_1h_1+v_2h_2+c_3).
$$
### Loss

For one example,
$$
L=\frac12(\hat y-y)^2.
$$
For the complete dataset,
$$
L= \frac{1}{N} \sum_i(\hat y_i-y_i)^2.
$$
### Optimization problem
$$
\theta^\star =
\arg\min_\theta L(\theta).
$$
### Backpropagation

Compute
$$
\frac{\partial L}{\partial p}
$$
for every parameter $p$.

### Gradient-descent update
$$
p \leftarrow
p-\eta\frac{\partial L}{\partial p}.
$$
Repeating this process allows the network to discover weights and biases
that transform the original data into a representation in which
classification becomes simple.

---
## 21. The whole idea in one sentence

> A neural network does not solve XOR merely by drawing two lines: its
> hidden neurons use nonlinear activation functions to transform the
> position of each input relative to those lines into new coordinates
>
$$
(h_1,h_2)
$$
, and in this transformed space the output neuron can
> linearly separate the two classes.

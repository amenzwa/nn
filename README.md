# nn

## _a simple neural networks implementation in C_

[TOC]

# INTRODUCTION

This project contains simple implementations of two well-known neural network algorithms: [David Rumelhart](https://en.wikipedia.org/wiki/David_Rumelhart)'s error back-propagation (EBP) supervised learning algorithm and [Teuvo Kohonen](https://en.wikipedia.org/wiki/Teuvo_Kohonen)'s self-organising map (SOM) unsupervised learning algorithm.

My goal is to show programmers how mathematics in scientific papers is transformed into prototype code. This is an essential skill for all programmers who work in science and engineering. As such, the code here is a direct translation of the mathematical descriptions given by Rumelhart in his paper [_Learning Internal Representation by Error Propagation_](https://www.gwern.net/docs/ai/nn/1986-rumelhart.pdf) (LIR) and by Kohonen in his paper [_The Self-Organising Map_](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf) (SOM). Do note that this project is a work in progress. I will be making regular updates to both the code and this description, until the project reaches its maturity.

To learn the theory of back-propagation neural networks, read LIR (about 30 pages) and SOM (about 20 pages). To understand the modern, vectorised implementation, read chapter 4 _Multilayer Feedforward Networks_ (about 70 pages) and chapter 7 _Matching and Self-Organizing Networks_ (about 60 pages) of [Jacek Zurada](https://en.wikipedia.org/wiki/Jacek_M._Zurada)'s classic textbook [_Introduction to Artificial Neural Systems_](https://www.amazon.com/Introduction-Artificial-Neural-Systems-Zurada/dp/053495460X) (ANS). LIR is written by a psychologist, SOM is written by a neuroscientist, and ANS is written by an electrical engineer. Studying these sources will give you a diverse perspective on this broad, multi-disciplinary subject.

# PRACTICAL

## *about this project*

This project is implemented in standard C, and it is self contained; it makes no references to third-party libraries. The EBP implementation includes XOR-2 and Encoder-8, two of the several example networks described by Rumelhart. The SOM implementation includes the minimum spanning tree (MST) example network described by Kohonen. For pedagogy and for clarity purposes, the code contains few error checks and even fewer code optimisations, because check and speed-up tricks obscure the algorithm. This intentional shoddiness is tolerable, since this code is meant for instructional use and not for production use.

An artificial neuron produces its output by summing up its weighted input values first, then transforming this net input with a non-linear activation function. The figure below shows the artificial neuron model. All practical neural network algorithms use this neuron arranged in layers. Some algorithms, like the SOM, use a single layer of neurons. But most algorithms use multiple layers of neurons, like the EBP.

![artificial neuron](./doc/NeuronMcCullochPitts.jpg)

LIR describes the EBP algorithm using the summation notation. For instance, the output of the neuron $n_j$ in the current layer is defined as $o_j = f(net_j)$. Here, $o_j$ is the output of the neuron $n_j$ in the current layer, $net_j$ is the net input to the neuron $n_j$, and $f$ is the activation function of the neuron $n_j$. The net input to the neuron $n_j$ is computed as $net_j = ∑_i w_{ji} o_i$, where $w_{ji}$ is the weight value of the connection between the neuron $n_j$ in the current layer and the neuron $n_i$ in the upstream layer and $o_i$ is the output of the neuron $n_i$.

In keeping with LIR description, `lir.c` implements the algorithm using `for` loops. Neural networks literature indexes neurons in layers using $i$, $j$, and $k$ for upstream (input-side), current, and downstream (output-side) layers. Hence, neural networks code accesses arrays as `a[j][i]`, as opposed to the traditional C convention `a[i][j]`. Programmers beware. In pseudocode, `o[l][j]` is computed like this:

```pseudocode
for p in P # data patterns
  for l in L # processing layers
    for j in N[l] # nodes in current layer l
      net = 0.0
      for i in N[l - 1] # nodes in upstream layer l - 1
        net += w[l][j][i]
      o[l][j] = f(net);
```

That is a whole lot of `for` loops. But it is an honest translation of Rumelhart's algorithm to C.

On the other hand, every implementation of back-propagation you will find on the Internet, be it a prototype or a production version, are vectorised versions. Indeed, the most popular machine learning framework, [TensorFlow](https://www.tensorflow.org/), is named so because it is implemented using [tensors](https://en.wikipedia.org/wiki/Tensor). Simply put, tensors are dimensional extensions of matrices, just like matrices are extensions of vectors and vectors are extensions of scalars.

In vectorised version, like that described in ANS, the data pattern is the `p` vector, the weights of the layer `l` is the `w[l]` matrix, the output of the layer `l` is the `o[l]` vector, and so on. Matrix-based implementation is well-suited to modern GPUs, which are equipped with powerful matrix manipulation pipelines (because transformations in [3D computer graphics](https://en.wikipedia.org/wiki/Transformation_matrix#Examples_in_3D_computer_graphics) are implemented using matrices). The matrix-based implementation is also far more compact and is easier to understand. The loopy pseudocode above reduces to the following vectorised pseudocode:

```pseudocode
for p in P
  for l in L
    o[l] = f(w[l] * p)
```

Despite all the advantages of the matrix-based EBP implementation, I chose the `for` loop version given in LIR, so as to show explicitly how the equations are realised in code. Unlike LIR, however, SOM presents its winner-takes-all competitive algorithm using the more common matrix notation. As such, I give here a vectorised implementation for SOM.

## _using the programme_

The project is structured thus:

```pseudocode
~/nn/
  LICENSE         # MIT license
  Makefile        # build script
  README.md       # this document
  bin/            # binaries directory
    test.sh       # test script
  csv.[ch]        # CSV utility
  dat/            # CSV data directory
    lir-enc8*.csv # encoder problem from LIR
    lir-xor2*.csv # XOR problem from LIR
    som-mst*.csv  # minimum spanning tree problem from SOM
    som-rgb*.csv  # RGB colour classification problem
  etc.[ch]        # network utilities
  lir.[ch]        # LIR implementation
  lirmain.c       # LIR main()
  som.[ch]        # SOM implementation
  sommain.c       # SOM main()
  vec.[ch]        # vector algebra utilities
```

The programmes are written for Unix-like operating systems. I do not program on Windows; neither should you. To compile and run the programmes, type in the following at a Unix command prompt:

```shell
$ cd ~/nn
$ make clean all
...
$ ./lir lir-xor2
...
$ ./lir lir-enc8
...
$ ./som som-mst
...
$ ./som som-rgb
...
```

Almost every statement in `lir.[ch]` and `som.[ch]` modules is commented. The comments cite LIR, SOM, and ANS by chapter, section, equation, and page, thus allowing you to trace the C functions back to their source equations. And to aid tracing, I named the network parameters as close as practicable to the respective author's notation.

The procedure `run()` in `*main.c` first loads from the `dat/` data directory the CSV configuration file of the specified network, say `dat/lir-xor2.csv`. This configuration file specifies the network architecture and the training parameters:

- LIR:

  - `name`—name of the network (also the base name of the CSV files)
  - `C`—number of training cycles
  - `L`—number of processing layers
  - `I`—number of input taps
  - `N`—number of nodes per processing layer, separated by `|`
  - `f`—layer-wide activation function (`...u` for unipolar; `...b` for bipolar)
  - `eta`—learning rate
  - `alpha`—momentum factor
  - `epsilon`—RMS error criterion
  - `P`—number of data patterns
  - `shuffle`—shuffle pattern presentation order

- SOM:
  - `name`—name of the network (also the base name of the CSV files)
  - `C`—number of training cycles
  - `I`—number of input taps
  - `W`—number of nodes in the $x$ direction
  - `H`—number of nodes in the $y$ direction
  - `dist`—distance measure (`inner` for inner product; `euclidean` for Euclidean distance)
  - `alpha`—learning factor
  - `epsilon`—RMS error criterion
  - `P`—number of data patterns
  - `shuffle`—shuffle pattern presentation order

Using these network parameters, `run()` creates a network, loads the pattern vectors, and train the network. During training, the current RMS error is reported every few cycles. Upon completion of training, `run()` prints out the final weights. The pattern vectors are specified in their respective CSV files, one row per pattern.

The module `etc.[ch]` implements utilities common to both EBP and SOM networks, such as the initial weights randomiser. This module also contains the various activation functions used by the EBP network. Each activation function has a unipolar version and a bipolar version.

The SOM network does not use activation functions; instead, it uses vector-space distance measures. The inner product (similarity cosine) measure is implemented by the `vecinner()` function and the Euclidean distance measure is implemented by the `veceuclidean()` function, which are defined in the `vec.[ch]` module. The module `vec.[ch]` implements vector and matrix manipulation utilities. Refer to chapter 7 _Vector Algebra_ and chapter 8 _Matrices and Vector Spaces_ of [_Mathematical Methods for Physics and Engineering_](https://www.amazon.com/Mathematical-Methods-Physics-Engineering-Comprehensive-ebook/dp/B00AKE1QJU), Riley (2006).

The module `csv.[ch]` implements a simple CSV parser described in section 4.1 _Comma-Separated Values_ of [_The Practice of Programming_](https://www.amazon.com/Practice-Programming-Addison-Wesley-Professional-Computing/dp/020161586X), Kernighan (1999).

## _a case for C_

I chose the [C programming language](<https://en.wikipedia.org/wiki/C_(programming_language)>) for these reasons. C is a small, simple imperative language, so there is little or no abstractions to distract us from our main purpose. C is also very close to hardware; it is but a thin coat of syntactic sugar atop assembly, so it is the fastest high-level language. Direct access to hardware, speed, and simplicity are why C became the canonical system programming language over the decades. Exploiting C's strengths and coping with its many traps makes the programmer more mechanically sympathetic, a trait modern programmers have lost long ago. Those who use modern, GPU-based deep learning frameworks will benefit from knowing C. Lastly, this year 2022, is C's 50th anniversary, and I wish to honour this long-lived language that is still blazing trails, despite its age. It is remarkable that C has change very little over the past five decades. This unrivalled stability is a testament to the far-reaching vision of its designer, [Dennis Ritchie](https://en.wikipedia.org/wiki/Dennis_Ritchie).

## _experimenting on your own_

If you want to experiment with your own EBP network, create `lir-yours.csv`, `lir-yours-i.csv`, and `lir-yours-t.csv` as described above, place the CSV files in the directory `./dat/`, and type `./lir lir-yours` at the command prompt. For SOM network, use `som-yours.csv` and `som-yours-i.csv`. The `lir-` and `som-` prefixes are there only to keep the CSV files organised by network architecture.

When creating your own data files for EBP networks, normalise the input and target vector components to the closed interval $[0.1, 0.9]$. As explained in LIR p. 9, the asymptotic nature of the sigmoid activation function prevents the network from ever reaching $0$ or $1$ saturated values. So, we keep the input values to EBP within this unsaturated range. On the other hand, SOM networks can cope with raw, unnormalised data, even RGB bitmap images.

Remember that both `lir` and `som` programmes in this project accepts only CSV-formatted UNIX text files. Given the minimal error checking in the code, non-CSV files, binary files, and Windows text files will likely crash the programmes.

# THEORETICAL

I present an informal description of EBP and SOM algorithms in this section. I give just enough explanation to enable you to navigate the code. If you want more details, see my article [_How Artificial Intelligence Works_](https://amenzwa.github.io/stem/AI/HowAIWorks/). Better still, read the original papers by Rumelhart and Kohonen.

## *error back-propagation*

EBP is a connectionist reimagining of the well known [gradient descent optimisation](https://en.wikipedia.org/wiki/Gradient_descent) technique invented by [Cauchy](https://en.wikipedia.org/wiki/Augustin-Louis_Cauchy) in 1847. The innovation that EBP brings is the efficient, parallel implementation, which is a common trait of neural network algorithms. By "parallel", I mean the parallelism within each layer of the network, which is exploited by all modern, GPU-based, vectorised implementations. Such optimisation techniques are beyond the scope, and contrary to the pedagogical purpose, of this project.

EBP, like all other neural networks, is constructed from artificial neurons arranged in layers, as shown in the figure below. A neuron is just a non-linear function that maps the weighted sum of its inputs to an output value, based on a bias value: $\mathbf{o} = \mathbf{f}(\mathbf{net})$. Here, $\mathbf{net} = \mathbf{W} \mathbf{i}$ is the net input values of the neurons in the layer, the $\mathbf{W}$ is the matrix that represents the weights on the connections entering the layer, $\mathbf{i}$ is the input vector to the layer, $\mathbf{o}$ is the output vector of the layer, and $\mathbf{f}$ is a vector-valued non-linear activation function with a bias $𝜃$. Adjusting the bias term translates the function along its input axes. This has the effect of making the neuron fire (produces its output) at a lower or a higher net input value. It is a common practice in EBP implementations to treat the bias $𝜃$ as though it were a component of the weight matrix. The neurons within a layer are independent of each other, but they do fire synchronously.

![error back-propagation neural network](./doc/ArchitectureBackpropagation.jpg)

EBP, like all neural networks, learns incrementally the structure and distribution of the patterns $\mathbf{p}$ drawn from the training set $P^n$, which is a subset of an $n$-dimensional vector space. During training, patter vectors are presented to the network, one by one, often in random order. Each layer of neurons transforms the input vector $\mathbf{i}$, which it receives from the adjacent upstream layer, into the output vector $\mathbf{o}$. The layer's output vector, in turn, is fed forward to the adjacent downstream layer. This part of the learning process is referred as the _forward pass_.

When the neuron responses reach the last (output) layer of the network, the output vector $\mathbf{o}$ is compared against the target vector $\mathbf{t}$, which represents the desired, correct output for a given data pattern $\mathbf{p}$. Since learning begins with randomised weights, the network's outputs in the early phase of the training are bound to be incorrect. That is, the error vector $\mathbf{𝛿}_o = \mathbf{t} - \mathbf{o}$ is non-zero. The $o$ subscript denotes the output layer. Note that each input-target pattern vector pair generates an error vector at the output layer.

This commences the _backward pass_ of the learning process. We use this output-layer error vector to compute that layer's weight adjustment matrix: $\Delta\mathbf{W} \leftarrow 𝜂 \mathbf{𝛿} \mathbf{i} + 𝛼 \Delta\mathbf{W}$, where $\Delta\mathbf{W}$ is the weight adjustment matrix, $𝜂$ is the learning rate, $\mathbf{𝛿}$ is the output-layer error vector as computed above, $\mathbf{i}$ is the input vector fed to the output layer, and $𝛼$ is the momentum factor. The learning rate determines the step size of the gradient descent, and the momentum factor damps out large swings in weight adjustments caused by larger learning rate choices.

Next, we propagate the output-layer error signal, which was caused by a training data pattern, backward through the network, one layer at a time. We shall denote the output layer's adjacent upstream layer with the subscript $h$. The interior of the network, like the layer $h$, is called a _hidden layer_ in EBP parlance. We compute the hidden layer's error vector like this: $\mathbf{𝛿}_h = \mathbf{f}^{\prime}(\mathbf{net}_h)\ \mathbf{W}_o^t \mathbf{𝛿}_o$, where $\mathbf{𝛿}_h$ is the output error vector $\mathbf{𝛿}_o$ back-propagated to the layer $h$, $\mathbf{f}^{\prime}$ is the derivative of the layer's activation function, $\mathbf{net}_h$ is the net input vector of the layer, and $\mathbf{W}_o^t$ is the transpose of the weight matrix $\mathbf{W}_o$ of the layer. Simply put, we compute the error vector of a hidden layer by propagating backward the adjacent downstream layer's error vector through the weights of that downstream layer. This is the converse of how the net input vector is computed during the forward pass. This "bass-akward" process is why we must transpose the weight matrix when we compute the hidden layers' error vectors.

Using the weight adjustment equation given above, we accumulate the $\Delta \mathbf{W}$ for all the layers, for each data pattern. We then adjust all the weights in the network simultaneously: $\mathbf{W} \leftarrow \mathbf{W} + \Delta \mathbf{W}$. This complete one _training cycle_. A typical EBP network requires repeating the training cycle about $10,000$ times or until the [root-mean square](https://en.wikipedia.org/wiki/Root_mean_square) (RMS) error over one cycle dips below a predefined error criterion.

## *self-organising map*

SOM is a connectionist version of the well known [*k*-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) technique. The innovation that SOM brings, not surprisingly, is the efficient, parallel implementation. The $k$-means algorithm is notoriously slow: it is [NP-hard](https://en.wikipedia.org/wiki/NP-hardness).

EBP is a simple algorithm. SOM is at least an order of magnitude simpler. SOM consists of one layer of neurons, called the _map_, as shown in the figure below. Being an unsupervised algorithm, it requires no target vectors. The neurons in the map are arranged either in a rectangular grid or a hexagonal grid. For simplicity, we shall consider a rectangular grid.

![self-organizing map neural network](./doc/ArchitectureSelfOrganisingMap.jpg)

The underlying principle of SOM is the idea of winner-takes-all. An input pattern vector $\mathbf{i}$ is fed to the neurons via their respective weights arranged as the matrix $\mathbf{W}$. First, we compute the Euclidean distance between the input vector $\mathbf{i}$ and the weight vector $\mathbf{w}_j$ of each neuron $j$ in the network: $d_j = \lVert \mathbf{i} - \mathbf{w}_j \rVert$. Then, we find the neuron whose weight vector is the closest to the input vector, and label it the winner. Since the algorithm is unsupervised, there are no target values with which to compare the neuron outputs and, hence, we are not interested in the neurons' output values.

In this way, each input pattern picks out a winning neuron, and the winner's weights $\mathbf{w}_w$ are adjusted so as to make them even closer to the input, next time round: $\mathbf{w}_w \leftarrow \mathbf{w}_w + 𝛼\ (\mathbf{i} - \mathbf{w}_w)$. Here, $0.0 < 𝛼 < 1.0$ is the learning factor and $\mathbf{i}$ is the input patter vector. The learning factor begins with a maximum, say $0.9$, and incrementally decreases after the _ordering phase_, which is usually the first $1,000$ learning cycles. We usually clamp the learning factor at a minimum of about $0.1$.

In a simple competitive learning network, we update only the winning neuron's weights. But in an SOM network, we deal with a neighbourhood $N$ of nodes surrounding the winner neuron. The radius of $N$ at the start of the learning process is quite large, around half the width of the map, and it shrinks after the ordering phase, down to a minimum radius of $1$ (neighbourhood of $3 \times 3$ neurons). Typically, we reduce the strength of weight adjustment within the neighbourhood exponentially so that the winner receives the largest adjustments and the farthest neighbours receive smallest adjustments.

Repeating this process for all the patterns in the data set completes one learning cycle. Usually, an SOM network requires about $100,000$ learning cycles, or until the RMS error criterion is reached.

# CONCLUSION

My goal for this project is to show IT programmers how to convert equations into lines of code, using a subject that is near and dear to them—AI programming.

The most effective way to learn AI programming and to be proficient practically is to study the AI algorithms and their theoretical foundations. The code shows you how the implementation works, but to understand the theory, you must read the original sources. Armed with the informal, theoretical overview given above, you might find tackling scientific publications somewhat easier. In any event, you must study on your own the theoretical concepts of AI; start with the classic publications listed below.

There is a trend among young STEMers to under estimate the value of old publications, mistakenly believing that old means irrelevant. If that were so, no one would hold Archimedes, Newton, or Einstein in high esteem. STEM textbooks can be categorised as modern, classic, and vintage. Most _modern_ texts strive to cover all the important, recent advances and applications, so their theoretical presentations necessary take a less prominent role. Modern texts are an excellent way to keep abreast with the latest developments in the field. Some _classic_ texts, especially those that were published just after the emergence of a groundbreaking idea, tend to be the best if one wishes to study that idea, in-depth. And given that they were published at the birth of an idea, their pages are not polluted with application examples and practice guides, so they are easier to read for novices interested in the underlying theory. But after about a century, all texts begin to show their age and they become _vintage_: their then-current examples become outdated and their stale notations grate modern sensibilities. Still, there are several vintage texts that are good-reads, much like Shakespeare still is. So, do not dismiss an old textbook by its publication date imprinted inside the front cover.

## _references_

- Theory:
  - [_Learning Internal Representations by Error Propagation_](https://www.gwern.net/docs/ai/nn/1986-rumelhart.pdf), Rumelhart (1986)
  - [_The Self-Organising Map_](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf), Kohonen (1990)
  - [_Introduction to Artificial Neural Systems_](https://www.amazon.com/Introduction-Artificial-Neural-Systems-Zurada/dp/053495460X), Zurada (1992)
  - [_Mathematical Methods for Physics and Engineering_](https://www.amazon.com/Mathematical-Methods-Physics-Engineering-Comprehensive-ebook/dp/B00AKE1QJU), Riley (2006)
- Practice:
  - [_The Practice of Programming_](https://www.amazon.com/Practice-Programming-Addison-Wesley-Professional-Computing/dp/020161586X), Kernighan (1999)
  - [_The C Programming Language_](https://www.amazon.com/Programming-Language-2nd-Brian-Kernighan/dp/0131103628/ref=sr_1_1?keywords=c+programming+language&qid=1664230233&qu=eyJxc2MiOiIzLjc0IiwicXNhIjoiMy4wMCIsInFzcCI6IjIuOTIifQ%3D%3D&sprefix=c+programmin%2Caps%2C56&sr=8-1), Kernighan (1989)

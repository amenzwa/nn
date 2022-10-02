# nn
## *a simple neural networks implementation in C*

This project contains simple implementations of two well-known neural network algorithms: [David Rumelhart](https://en.wikipedia.org/wiki/David_Rumelhart)'s error back-propagation (EBP) supervised learning algorithm and [Teuvo Kohonen](https://en.wikipedia.org/wiki/Teuvo_Kohonen)'s self-organising map (SOM) unsupervised learning algorithm.

The goal here is to show programmers how mathematics in scientific papers is transformed into prototype code. As such, the implementations given here are direct translations of the mathematical descriptions given by Rumelhart in his paper [*Learning Internal Representation by Error Propagation*](https://www.gwern.net/docs/ai/nn/1986-rumelhart.pdf) (LIR) and by Kohonen in his paper [*The Self-Organising Map*](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf) (SOM).

Do note that this project is a work in progress. Once it has matured somewhat, I will post a detailed article on interpreting peer-reviewed scientific papers, translating mathematics into prototype code, and system programming concepts on [my GitHub Pages blog](https://amenzwa.github.io). In the meantime, see my article [*How Artificial Intelligence Works*](https://amenzwa.github.io/stem/AI/HowAIWorks/) for a high-level overview and a historical background on neural networks in particular and AI in general.

To learn the theory of back-propagation neural networks, read LIR (about 30 pages) and SOM (about 20 pages). To understand the modern, vectorised implementation, read chapter 4 *Multilayer Feedforward Networks* (about 70 pages) and chapter 7 *Matching and Self-Organizing Networks* (about 60 pages) of [Jacek Zurada](https://en.wikipedia.org/wiki/Jacek_M._Zurada)'s classic textbook [*Introduction to Artificial Neural Systems*](https://www.amazon.com/Introduction-Artificial-Neural-Systems-Zurada/dp/053495460X) (ANS). LIR is written by a psychologist, SOM is written by a neuroscientist, and ANS is written by an electrical engineer. Studying these sources will give you a diverse perspective on this broad, multi-disciplinary subject.

There is a trend among young STEMers to under estimate the value of old publications, mistakenly believing that old means irrelevant. If that were so, no one would hold Archimedes, Newton, or Einstein in high esteem. STEM textbooks can be categorised as modern, classic, and vintage. Most *modern* texts strive to cover all the important, recent advances and applications, so their theoretical presentations necessary take a less prominent role. Modern texts are an excellent way to keep abreast with the latest developments in the field. Some *classic* texts, especially those that were published just after the emergence of a groundbreaking idea, tend to be the best if one wishes to study that idea, in-depth. And given that they were published at the birth of an idea, their pages are not polluted with application examples and practice guides, so they are easier to read for novices interested in the underlying theory. But after about a century, all texts begin to show their age and they become *vintage*: their then-current examples become outdated and their stale notations grate modern sensibilities. Still, there are several vintage texts that are good-reads, much like Shakespeare still is. So, do not dismiss an old textbook by its publication date imprinted inside the front cover.

## *about this project*

This project is implemented in standard C. The EBP implementation includes XOR-2 and Encoder-8, two of the several example networks given by Rumelhart. The SOM implementation includes the minimum spanning tree example given by Kohonen. The goal of these programmes is to show programmers how to convert mathematics into prototype code, a skill essential for all programmers who work in science and engineering fields. For pedagogy and for clarity purposes, the code contains few error checks and even fewer code optimisations, because check and speed-up infrastructure code obscure the algorithm. This intentional shoddiness is tolerable, since this code is not intended to be used in production.

LIR describes the algorithm using summations. For instance, the output of the neuron $n_j$ in the current layer is defined as $o_j = f(net_j)$. Here, $o_j$ is the output of the neuron $n_j$, $net_j$ is the net input to the neuron $n_j$, and $f$ is the activation function of the neuron $n_j$. The net input to the neuron $n_j$ is computed as $net_j = ∑_i w_{ji} o_i$, where $w_{ji}$ is the weight value of the connection between the neuron $n_j$ in the current layer and the neuron $n_i$ in the upstream layer and $o_i$ is the output of the neuron $n_i$.

In keeping with LIR description, `lir.c` implements the algorithm using `for` loops. Neural networks literature uses indices $i$, $j$, and $k$ for upstream, current, and downstream layers. Hence, neural networks code accesses arrays as `a[j][i]`, as opposed to the traditional C convention `a[i][j]`. Programmers beware. In pseudocode, `o[l][j]` is computed like this:

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

On the other hand, every implementation of back-propagation you will find on the Internet, be it a prototype or a production version, will use vector algebra. Indeed, the most popular deep learning framework, [TensorFlow](https://www.tensorflow.org/), is named so because it is implemented using [tensors](https://en.wikipedia.org/wiki/Tensor). Simply put, tensors are dimensional extensions of matrices, just like matrices are extensions of vectors and vectors are extensions of scalars.

In vectorised version, like that described in ANS, the data pattern is the `p` vector, the weights of the layer `l` is the `w[l]` matrix, the output of the layer `l` is the `o[l]` vector, and so on. Matrix-based implementation is well-suited to modern GPUs, which are equipped with powerful matrix manipulation pipelines (because transformations in [3D computer graphics](https://en.wikipedia.org/wiki/Transformation_matrix#Examples_in_3D_computer_graphics) are implemented using matrices). The matrix-based implementation is also far more compact and is easier to understand. The loopy pseudocode above reduces to the following vectorised pseudocode:

```pseudocode
for p in P
  for l in L
    o[l] = f(w[l] * p)
```

Despite all the advantages of the matrix-based EBP implementation, I chose the `for` loop version given in LIR, so as to show explicitly how the equations are realised in code. Unlike LIR, through, SOM presents its algorithm in matrix notation. As such, I give here a vectorised implementation for SOM.

## *using the programme*

The project is structured thus:

```pseudocode
~/nn/
  Makefile        # build script
  README.md       # this document
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
  - `arch`—network architecture (`r4` for 4-neighbour rectangular; `r8` for 8-neighbour rectangular)
  - `dist`—distance measure (`inner` for inner product; `euclidean` for Euclidean distance)
  - `alpha`—learning factor
  - `epsilon`—RMS error criterion
  - `P`—number of data patterns
  - `shuffle`—shuffle pattern presentation order


Using these network parameters, `run()` creates a network, loads the pattern vectors, and train the network. During training, the current RMS error is reported every few cycles. Upon completion of training, `run()` prints out the final weights. The pattern vectors are specified in their respective CSV files, one row per pattern.

The module `etc.[ch]` implements utilities common to both EBP and SOM networks, such as the initial weights randomiser. This module also contains the various activation functions used by the EBP network. Each activation function has a unipolar version and a bipolar version.

The SOM network does not use activation functions; instead, it uses vector-space distance measures. The inner product (similarity cosine) measure is implemented by the `vecinner()` function and the Euclidean distance measure is implemented by the `veceuclidean()` function, which are defined in the `vec.[ch]` module. The module `vec.[ch]` implements vector and matrix manipulation utilities. Refer to chapter 7 *Vector Algebra* and chapter 8 *Matrices and Vector Spaces* of [*Mathematical Methods for Physics and Engineering*](https://www.amazon.com/Mathematical-Methods-Physics-Engineering-Comprehensive-ebook/dp/B00AKE1QJU), Riley (2006).

The module `csv.[ch]` implements a simple CSV parser described in section 4.1 *Comma-Separated Values* of [*The Practice of Programming*](https://www.amazon.com/Practice-Programming-Addison-Wesley-Professional-Computing/dp/020161586X), Kernighan (1999).

## *experimenting on your own*

If you want to experiment with your own EBP network, create `lir-yours.csv`, `lir-yours-i.csv`, and `lir-yours-t.csv` as described above, place the CSV files in the directory `./dat/`, and type `./lir lir-yours` at the command prompt. For SOM network, use `som-yours.csv` and `som-yours-i.csv`. The `lir-` and `som-` prefixes are there only to keep the CSV files organised by network architecture.

When creating your own data files for EBP networks, normalise the input and target vector components to the closed interval $[0.1, 0.9]$. As explained in LIR p. 9, the asymptotic nature of the sigmoid activation function prevents the network from ever reaching $0$ or $1$ saturated values. So, we keep the input values to EBP within this unsaturated range. On the other hand, SOM networks can cope with raw, unnormalised data, even RGB bitmap images.

Remember that both `lir` and `som` programmes in this project accepts only CSV-formatted UNIX text files. Given the minimal error checking in the code, non-CSV files, binary files, and Windows text files will likely crash the programmes.

## *a case for C*

I chose the [C programming language](https://en.wikipedia.org/wiki/C_(programming_language)) for these reasons. C is a small, simple imperative language, so there is little or no abstractions to distract us from our main purpose. C is also very close to hardware; it is but a thin coat of syntactic sugar atop assembly, so it is the fastest high-level language. Direct access to hardware, speed, and simplicity are why C became the canonical system programming language over the decades. Exploiting C's strengths and coping with its many traps makes the programmer more mechanically sympathetic, a trait modern programmers have lost long ago. Those who use modern, GPU-based deep learning frameworks will benefit from knowing C. Lastly, this year 2022, is C's 50th anniversary, and I wish to honour this long-lived language that is still blazing trails, despite its age. It is remarkable that C has change very little over the past five decades. This unrivalled stability is a testament to the far-reaching vision of its designer, [Dennis Ritchie](https://en.wikipedia.org/wiki/Dennis_Ritchie).

## *references*

- [*Learning Internal Representations by Error Propagation*](https://www.gwern.net/docs/ai/nn/1986-rumelhart.pdf), Rumelhart (1986)
- [*The Self-Organising Map*](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf), Kohonen (1990)
- [*Introduction to Artificial Neural Systems*](https://www.amazon.com/Introduction-Artificial-Neural-Systems-Zurada/dp/053495460X), Zurada (1992)
- [*Mathematical Methods for Physics and Engineering*](https://www.amazon.com/Mathematical-Methods-Physics-Engineering-Comprehensive-ebook/dp/B00AKE1QJU), Riley (2006)
- [*The Practice of Programming*](https://www.amazon.com/Practice-Programming-Addison-Wesley-Professional-Computing/dp/020161586X), Kernighan (1999)
- [*The C Programming Language*](https://www.amazon.com/Programming-Language-2nd-Brian-Kernighan/dp/0131103628/ref=sr_1_1?keywords=c+programming+language&qid=1664230233&qu=eyJxc2MiOiIzLjc0IiwicXNhIjoiMy4wMCIsInFzcCI6IjIuOTIifQ%3D%3D&sprefix=c+programmin%2Caps%2C56&sr=8-1), Kernighan (1989)

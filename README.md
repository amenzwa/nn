# bp
## *a simple back-propagation implementation in C*

This is a simple implementation of [David Rumelhart](https://en.wikipedia.org/wiki/David_Rumelhart)'s back-propagation neural network algorithm. It is a direct implementation of the proof given by Rumelhart in his seminal paper [*Learning Internal Representation by Error Propagation*](https://www.gwern.net/docs/ai/nn/1986-rumelhart.pdf) (LIR). A modern, matrix-based implementation is described by [Jacek Zurada](https://en.wikipedia.org/wiki/Jacek_M._Zurada) in his classic textbook [*Introduction to Artificial Neural Systems*](https://www.amazon.com/Introduction-Artificial-Neural-Systems-Zurada/dp/053495460X) (ANS). To understand the theory, read LIR in its entirety (about 30 pages). To understand the matrix-based implementation, read chapter 4 of ANS (about 70 pages).

Do note that this still is a work-in-progress project. Once this project has matured somewhat, I will post a detailed article on reading peer-reviewed scientific papers, translating mathematics into prototype code, and system programming concepts on [my GitHub Pages blog](https://amenzwa.github.io). Please be patient.

This project is implemented in standard C, and includes XOR-2 and Encoder-8, two of the several example networks given in LIR. The goal of these projects is to show programmers how to convert mathematics into prototype code, a skill essential for all programmers who work in science and engineering fields. For pedagogical purpose and for clarity, the code includes few error checks, because error checks obscure the logic. This intentional shoddiness is tolerable, since this code is not intended to be used in production.

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

That is a whole lot of `for` loops. But it is an honest translation of Rumelhart's algorithm to C. A more common implementation technique, however, is to use vector algebra. The data pattern is the `p` vector, the weights of the layer `l` is the `w[l]` matrix, the output of the layer `l` is the `o[l]` vector, and so on. Because vectorised implementation is well-suited to modern GPUs, which are equipped with powerful matrix manipulation pipeline, all modern implementations of the back-propagation algorithm use matrices. The matrix-based implementation is also far more compact and is easier to understand. The loopy pseudocode above reduces to the following vectorised pseudocode:

```pseudocode
for p in P
  for l in L
    o[l] = f(w[l] * p)
```

I will publish a matrix-based version of this project in the near future.

## *using the programme*

The project is structured thus:

```pseudocode
~/bp/
  csv.[ch]
  dat/
    *.csv
  lir.[ch]
  lirmain.c
```

The programme is written for Unix-like operating system. I do not work on Windows; neither should you. To compile and run `lir`, type in the following at a Unix command prompt:

```shell
$ cd ~/bp
$ make clean all
...
$ ./lir xor2
...
$ ./lir enc8
...
```

Almost every statement in `lir.[ch]` module is commented. The comments cite LIR and ANS by chapter, section, equation, and page, thus allowing you to trace the C functions back to their source equations.

The procedure `run()` first loads from the `dat/` directory the CSV configuration file of the specified network, say `dat/xor2.csv`. This configuration file specifies the network architecture and training parameters:

- `C`—number of training cycles
- `L`—number of processing layers
- `eta`—learning rate
- `alpha`—momentum factor
- `epsilon`—RMS error criterion
- `P`—number of data patterns
- `I`—number of input taps
- `N`—number of nodes per processing layer, separated by `|`
- `f`—layer-wide activation function (`...u` for unipolar, `...b` for bipolar)

Using these parameters, `run()` creates a network, loads the input and target pattern vectors from `./dat/...-i.csv` and `./dat/...-t.csv`, and train the network. During training, the current RMS error is reported every few cycles. Upon completion of training, `run()` prints out the recall RMS error and the final weights. The input and target vectors are specified in their respective CSV files, one row per pattern.

The module `csv.[ch]` implements a simple CSV parser described in section 4.1 *Comma-Separated Values* of [*The Practice of Programming*](https://www.amazon.com/Practice-Programming-Addison-Wesley-Professional-Computing/dp/020161586X), Kernighan (1999).

## *experimenting on your own*

If you want to experiment with your own network, create `yours.csv`, `yours-i.csv`, and `yours-t.csv` as described above, place the CSV files in the directory `./dat/`, and type `./lir yours` at the command prompt.

When creating your own data files, normalise the input and target vector components to the closed interval $[0.1, 0.9]$. As explained in LIR p. 9, the asymptotic nature of the sigmoid activation function prevents the network from ever reaching $0$ or $1$ saturated values. In compensation, we keep the values within this unsaturated range.

## *a case for C*

I chose the [C programming language](https://en.wikipedia.org/wiki/C_(programming_language)) for these reasons. C is a small, simple imperative language, so there is little or no abstractions to distract us from our main purpose. C is also very close to hardware; it is but a thin coat of syntactic sugar atop assembly, so it is the fastest high-level language. Direct access to hardware, speed, and simplicity are why C became the canonical system programming language over the decades. Exploiting C's strengths and coping with its many traps makes the programmer more mechanically sympathetic, a trait modern programmers have lost long ago. Those who use modern, GPU-based deep learning frameworks will benefit from knowing C. Lastly, this year 2022, is C's 50th anniversary, and I wish to honour this long-lived language that is still blazing trails, despite its age. It is remarkable that C has change very little over the past five decades. This stability is a testament to its original designer, [Dennis Ritchie](https://en.wikipedia.org/wiki/Dennis_Ritchie).

## *references*

- [*Learning Internal Representations by Error Propagation*](https://www.gwern.net/docs/ai/nn/1986-rumelhart.pdf), Rumelhart (1986)
- [*Introduction to Artificial Neural Systems*](https://www.amazon.com/Introduction-Artificial-Neural-Systems-Zurada/dp/053495460X), Zurada (1992)
- [*The Practice of Programming*](https://www.amazon.com/Practice-Programming-Addison-Wesley-Professional-Computing/dp/020161586X), Kernighan (1999)
- [*The C Programming Language*](https://www.amazon.com/Programming-Language-2nd-Brian-Kernighan/dp/0131103628/ref=sr_1_1?keywords=c+programming+language&qid=1664230233&qu=eyJxc2MiOiIzLjc0IiwicXNhIjoiMy4wMCIsInFzcCI6IjIuOTIifQ%3D%3D&sprefix=c+programmin%2Caps%2C56&sr=8-1), Kernighan (1989)

# Embedded Binarized Neural Networks
    
We study embedded Binarized Neural Networks (eBNNs) with the aim of allowing current binarized neural networks (BNNs) in the literature to perform feedforward inference efficiently on small embedded devices. We focus on minimizing the required memory footprint, given that these devices often have memory as small as tens of kilobytes (KB). Beyond minimizing the memory required to store weights, as in a BNN, we show that it is essential to minimize the memory used for temporaries which hold intermediate results between layers in feedforward inference. To accomplish this, eBNN reorders the computation of inference while preserving the original BNN structure, and uses just a single floating-point temporary for the entire neural network. All intermediate results from a layer are stored as binary values, as opposed to floating-points used in current BNN implementations, leading to a 32x reduction in required temporary space. We provide empirical evidence that our proposed eBNN approach allows efficient inference (10s of ms) on devices with severely limited memory (10s of KB). For example, eBNN achieves 95% accuracy on the MNIST dataset running on an Intel Curie with only 15 KB of usable memory with an inference runtime of under 50 ms per sample.

This repository contains a code to train a neural network and generate C/Arduino code for embedded devices such as Arduino 101. 

## Setup
Clone and enter the repoistory.
```bash
git clone git@github.com:kunglab/ebnn.git
cd ebnn
```
Setup virtualenv ([read more here if interested](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/)).
```bash
virtualenv env
source env/bin/activate
```
Now the bash prompt should have *(env)* in front of it. 

Install required packages. Note: This project uses Chainer 2.0, which has a [few differences](http://docs.chainer.org/en/stable/upgrade.html) to Chainer 1.0.
```bash
pip install -r requirements.txt
```

## Quick Start
This library has two components: a python module that trains the eBNN and generates a C header file, and the C library which uses the generated header file and is compiled on the target device to perform inference. A simple example of network training is located at located in [examples/simple.py](https://github.com/kunglab/ebnn/blob/master/examples/simple.py).

This will generate the simple.h header file which requires the ebnn.h file. These two files should be included in the C/Arduino code. The C library is used as follows: 

```c
#include <stdio.h>
#include <stdint.h>
#include "simple.h"

int main()
{
  float input[28*28];
  uint8_t output[1];
   
  //simulate a 28 by 28 greyscale image
  for(int i=0; i < 28*28; ++i) {
    input[i] = i;
  }
    
  ebnn_compute(input, output);
  printf("%d\n", output[0]);
   
  return 0;
}
```

For examples of generated networks, see the [c/tests](https://github.com/kunglab/ebnn/tree/master/c/tests) folder.

## Paper

Our paper is available [here](http://www.eecs.harvard.edu/~htk/publication/2017-ewsn-mcdanel-teerapittayanon-kung.pdf)

If you use this model or codebase, please cite:
```bibtex
@article{mcdanelembedded,
  title={Embedded Binarized Neural Networks},
  author={McDanel, Bradley and Teerapittayanon, Surat and Kung, HT},
  jornal={Proceedings of the 2017 International Conference on Embedded Wireless Systems and Networks},
  year={2017}
}
```

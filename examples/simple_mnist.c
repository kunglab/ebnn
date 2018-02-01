#include <stdio.h>
#include <stdint.h>
#include "simple_mnist.h"
#include "mnist_data.h"

int main()
{
  uint8_t output[1];
   
  for(int j = 0; j < 20; ++j) {
    ebnn_compute(&train_data[1*28*28*j], output);
    printf("actual: %d, predicted: %d\n", (int)train_labels[j], output[0]);
  }
   
  return 0;
}

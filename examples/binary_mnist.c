#include <stdio.h>
#include <stdint.h>
#include "ebnn.h"
#include "simple_binary_mnist.h"
#include "binary_mnist_data.h"

int main()
{
  uint8_t output[1];
   
  int idx = 0;
  for(int j = 0; j < 20; ++j) {
    int index = 1*CEIL_POS((28*28)/8)*j; //Each element packed, since binary
    ebnn_compute(&train_data[index], output);
    printf("actual: %d, predicted: %d\n", (int)train_labels[j], output[0]);
  }
   
  return 0;
}

#include <stdio.h>
#include <stdint.h>
#include "ebnn.h"
#include "alex.h"
#include "mnist.h"

int main()
{
  uint8_t output[1];
   
  int idx = 0;
  for(int j = 0; j < 20; ++j) {
    ebnn_compute(&train_data[784*j], output);
    printf("actual: %d, predicted: %d\n", (int)train_labels[j], output[0]);
  }

  return 0;
}

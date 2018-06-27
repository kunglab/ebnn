#include <stdio.h>
#include <stdint.h>
#include "ebnn.h"
#include "alex.h"
#include "mnist_data.h"

int main()
{
  uint8_t output[1];
   
  int idx = 0;
  for(int j = 0; j < 20; ++j) {
    ebnn_compute(&train_data[784*j], output);
    int fail =  (int)train_labels[j] != output[0];
    printf("actual: %d %s predicted: %d\n", (int)train_labels[j], (fail ? "<>" : "=="), output[0]);
  }

  return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef NX
#  define NX 8000
#endif

#ifndef NY
#  define NY 8000
#endif

#ifndef BX
#  define BX 100
#endif

#ifndef BY
#  define BY 100
#endif

#define BT 1

float data[2][NY][NX];

void init()
{
  int x, y;
  int cx = NX/2, cy = 0; /* center of ink */
  int rad = (NX+NY)/8; /* radius of ink */


#pragma omp parallel for private(x) //collapse(2)
  for(y = 0; y < NY; y++) {
    for(x = 0; x < NX; x++) {

//      data[0][y][x] = rand();
//      data[1][y][x] = rand();

      float v = 0.0;
      if (((x-cx)*(x-cx)+(y-cy)*(y-cy)) < rad*rad && y>0) {
        v = 1.0;
      }
      data[0][y][x] = v;
      data[1][y][x] = v;
    }
  }
  return;
}

// Endo modify
#define KERNEL(AR, X, Y)                        \
    (((AR)[(Y)-1][(X)-1]                        \
      + (AR)[(Y)-1][(X)]                        \
      + (AR)[(Y)-1][(X)+1]                      \
      + (AR)[(Y)][(X)-1]                        \
      + (AR)[(Y)][(X)]                          \
      + (AR)[(Y)][(X)+1]                        \
      + (AR)[(Y)+1][(X)-1]                      \
      + (AR)[(Y)+1][(X)]                        \
      + (AR)[(Y)+1][(X)+1])/9.0)

void calc(int nt)
{
  int xo_split_size = NX/BX;
  int yo_split_size = NY/BY;
  
#pragma omp parallel
  {
    int xo, yo;
    int to; // to must be private
    //float tmp[2][BY+2*BT][BX+2*BT];

    for(to = 0; to < nt/BT; to++){
#pragma omp master
      {
        printf("(%d/%d) ", to*BT, nt);
        fflush(0);
      }
      
      int from_data = to%2;
      int to_data  = (to+1)%2;

#pragma omp for collapse(2) // yo loop and xo loop are parallelized
      for(yo = 0; yo < yo_split_size; yo++){
        for(xo = 0; xo < xo_split_size; xo++){
          int xi, yi;
          int x_start = xo*BX;
          int x_end = (xo+1)*BX;
          int y_start = yo*BY;
          int y_end = (yo+1)*BY;
          
          if (x_start < 1) x_start = 1;
          if (x_end > NX-1) x_end = NX-1;
          if (y_start < 1) y_start = 1;
          if (y_end > NY-1) y_end = NY-1;
          
          for (yi = y_start; yi < y_end; yi++) {
            for (xi = x_start; xi < x_end; xi++) {
              data[to_data][yi][xi] = KERNEL(data[from_data], xi, yi);
            }
          }
          
          // computation of a spatial block finished
        } // xo
      } // yo
    }
  } // to
  printf("\n");
  return;
}

int main(int argc, char **argv) {
  int i, j;
  struct timespec ts1, ts2, ts3, ts4, ts5;
  double init_time, calc_time_1,c2,c3;
  int nt = 100;

  printf("NX=%d, NY=%d, BX=%d, BY=%d, BT=%d\n",
         NX, NY, BX, BY, BT);
  
  clock_gettime(CLOCK_REALTIME, &ts1);

  init();
  
  clock_gettime(CLOCK_REALTIME, &ts2);

  calc(nt);

  clock_gettime(CLOCK_REALTIME, &ts3);
/*  calc(nt);
  clock_gettime(CLOCK_REALTIME, &ts4);
  calc(nt);
  clock_gettime(CLOCK_REALTIME, &ts5);
*/
  init_time = (double)(ts2.tv_sec - ts1.tv_sec) + ((double)(ts2.tv_nsec - ts1.tv_nsec))*1.e-9;
  calc_time_1 = (double)(ts3.tv_sec - ts2.tv_sec) + ((double)(ts3.tv_nsec - ts2.tv_nsec))*1.e-9;
//  c2 = (double)(ts4.tv_sec - ts3.tv_sec) + ((double)(ts4.tv_nsec - ts3.tv_nsec))*1.e-9;
//  c3 = (double)(ts5.tv_sec - ts4.tv_sec) + ((double)(ts5.tv_nsec - ts4.tv_nsec))*1.e-9;
//  printf("init time %f\n", init_time);
  printf("calc time %lf sec\n", calc_time_1);
  double gflops = (9.0 * NX * NY * nt) / calc_time_1 * 1.e-9; // 9 ops per point
  printf("speed %lf GFlops\n", gflops);
  //  printf("calc time %f\n", c2);
//  printf("calc time %f\n", c3);

}

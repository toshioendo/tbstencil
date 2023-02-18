// originally developed by Hiroki Aikawa
// reengineered by Toshio Endo
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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

#ifndef BT
#  ifdef time_block_size
#    define BT time_block_size
#  else
#    define BT 20
#  endif
#endif

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

// -------------
// |app|acp|anp|
// -------------
// |apc|acc|anc|
// -------------
// |apn|acn|ann|
// -------------
#define KERNELD(app, acp, anp, apc, acc, anc, apn, acn, ann)    \
  ((app)+(acp)+(anp)+(apc)+(acc)+(anc)+(apn)+(acn)+(ann)/9.0)

#define KERNEL(AR, X, Y)                                            \
  KERNELD((AR)[(Y)-1][(X)-1], (AR)[(Y)-1][(X)], (AR)[(Y)-1][(X)+1], \
          (AR)[(Y)  ][(X)-1], (AR)[(Y)  ][(X)], (AR)[(Y)  ][(X)+1], \
          (AR)[(Y)+1][(X)-1], (AR)[(Y)+1][(X)], (AR)[(Y)+1][(X)+1])

#define COMPLINE(DAR, DX, DY, SAR, SX, SY, LEN) \
  for (int ii = 0; ii < (LEN); ii++) {          \
    DAR[DY][(DX)+ii] = KERNEL(SAR, (SX)+ii, SY);   \
  } \

void calc(int nt)
{
#pragma omp parallel
  {
    int xo, yo;
    int to; // to must be private
    float tmp[2][BY+2*BT][BX+2*BT]; // 2 tmp buffers per thread
#pragma omp master
    printf("sizeof(tmp)=%ld per thread\n", sizeof(tmp));

    for(to = 0; to < nt/BT; to++){
#pragma omp master
      {
        printf("t=(%d/%d) ", to*BT, nt);
        fflush(0);
      }
      
      int from_data = to%2;
      int to_data  = (to+1)%2;

#pragma omp for collapse(2) // yo loop and xo loop are parallelized
      for(yo = 0; yo < NY; yo += BY){
        for(xo = 0; xo < NX; xo += BX){
          int xi, yi, ti;

          // [xs,xe)*[ys,ye) at (to+1)*BT should be computed
          int xs = xo;
          int ys = yo;
          int xe = xo+BX;
          int ye = yo+BY;
          // clip
          if (xs < 1) xs = 1;
          if (ys < 1) ys = 1;
          if (xe > NX-1) xe = NX-1;
          if (ye > NY-1) ye = NY-1;
          //printf("[%d,%d)*[%d,%d)\n", xs,ys,xe,ye);
          // offsets: data[*][x][y] corresponds to tmp[*][x-xoff][y-yoff]
          int xoff = xo-BT;
          int yoff = yo-BT;

          for (ti = 0; ti < BT; ti++) {
            // [xis,yis)*[xie,yie) is computed this time
            int xis = xs-(BT-1);
            int yis = ys-(BT-1);
            int xie = xe+(BT-1);
            int yie = ye+(BT-1);
            // clip
            if (xis < 1) xis = 1;
            if (yis < 1) yis = 1;
            if (xie > NX-1) xie = NX-1;
            if (yie > NY-1) yie = NY-1;

            if (BT == 1) {
              // no temporal blocking: data <= data
              //if (xo+yo==0) printf("data[%d] <= data[%d]\n", to_data, from_data);
              for (yi = yis; yi < yie; yi++) {
                COMPLINE(data[to_data], xis, yi, data[from_data], xis, yi, xie-xis);
              }
            }
            else if (ti == 0) {
              // the first inner time step: tmp <= data
              //if (xo+yo==0) printf("tmp[%d] <= data[%d]\n", 1, from_data);
              for (yi = yis; yi < yie; yi++) {
                int yi_tmp = yi-yoff;
                int xis_tmp = xis-xoff;
                COMPLINE(tmp[1], xis_tmp, yi_tmp, data[from_data], xis, yi, xie-xis);
              }
            }
            else if (ti < BT-1) {
              // middle inner time steps; tmp <=> tmp
              int from_tmp=ti%2;
              int to_tmp=(ti+1)%2;
              //if (xo+yo==0) printf("tmp[%d] <= tmp[%d]\n", to_tmp, from_tmp);
              for (yi = yis; yi < yie; yi++) {
                int yi_tmp = yi-yoff;
                int xis_tmp = xis-xoff;
                COMPLINE(tmp[to_tmp], xis_tmp, yi_tmp, tmp[from_tmp], xis_tmp, yi_tmp, xie-xis);
              }
            }
            else {
              // the last inner time step; data <= tmp
              assert(ti == BT-1);
              int from_tmp=ti%2;
              //if (xo+yo==0) printf("data[%d] <= tmp[%d]\n", to_data, from_tmp);
              for (yi = yis; yi < yie; yi++) {
                int yi_tmp = yi-yoff;
                int xis_tmp = xis-xoff;
                COMPLINE(data[to_data], xis, yi, tmp[from_tmp], xis_tmp, yi_tmp, xie-xis);
              }
            }
          }
          
        } // xo
      } // yo
    }
  } // to
  printf("\n");
  return;
}

int main(int argc, char **argv) {
  int i, j;
  struct timespec ts1, ts2, ts3;
  double init_time, calc_time;
  int nt = 100;

  printf("NX=%d, NY=%d, BX=%d, BY=%d, BT=%d\n",
         NX, NY, BX, BY, BT);
  
  clock_gettime(CLOCK_REALTIME, &ts1);

  init();
  
  clock_gettime(CLOCK_REALTIME, &ts2);

  calc(nt);

  clock_gettime(CLOCK_REALTIME, &ts3);

  init_time = (double)(ts2.tv_sec - ts1.tv_sec) + ((double)(ts2.tv_nsec - ts1.tv_nsec))*1.e-9;
  calc_time = (double)(ts3.tv_sec - ts2.tv_sec) + ((double)(ts3.tv_nsec - ts2.tv_nsec))*1.e-9;
//  printf("init time %f\n", init_time);
  printf("calc time %.3lf sec\n", calc_time);
  double GFlops = (9.0 * NX * NY * nt) / calc_time * 1.e-9; // 9 ops per point
  double GBps = (2.0*sizeof(float)*NX * NY * nt) / calc_time * 1.e-9;
  printf("speed %.3lf GFlops, %.3lf GB/s\n", GFlops, GBps);
}

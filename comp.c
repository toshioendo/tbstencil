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
    float tmp[2][BY+2*BT][BX+2*BT];
#pragma omp master
    printf("sizeof(tmp)=%ld per thread\n", sizeof(tmp));

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
          int xi, yi, ti;
          
          // computation a spatial block begins
          int from_tmp,to_tmp;
          if(yo == 0){
            int y_start = 0;
            int y_end = BY+BT;
            
            if(xo == 0){
              int x_start = 0;
              int x_end = BX+BT;
              
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x =xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug

              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;

                for (yi = y_start+1; yi < y_end-1-ti; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1; xi < x_end-1-ti; xi++) {
                    int tmp_x =xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = 1; yi < BY; yi++) {
                int tmp_y = yi;
                for (xi = 1; xi < BX; xi++) {
                  int tmp_x =xi;
                  data[to_data][yi][xi] = KERNEL(tmp[to_tmp], tmp_x, tmp_y);
                }
              }
            }
            ///
            else if(xo == xo_split_size-1){
              int x_start = xo*BX-BT;
              int x_end = NX;
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x =xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug
              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;
                for (yi = y_start+1; yi < y_end-1-ti; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1+ti; xi < x_end-1; xi++) {
                    int tmp_x =xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = 1; yi < BY; yi++) {
                int tmp_y = yi;
                int y_return = y_start+yi;
                for (xi = BT; xi < BX+BT; xi++) {
                  int tmp_x = xi;
                  int x_return = x_start+xi;
                  data[to_data][y_return][x_return] = KERNEL(tmp[to_tmp], tmp_x, tmp_y);
                }
              }
            }
            ///
            else{
              int x_start = xo*BX-BT;
              int x_end = (xo+1)*BX+BT;
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x = xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug
              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;
                for (yi = y_start+1; yi < y_end-1-ti; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1+ti; xi < x_end-1-ti; xi++) {
                    int tmp_x = xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = 1; yi < BY; yi++) {
                int tmp_y = yi;
                int y_return = y_start+yi;
                for (xi = BT; xi < BX+BT; xi++) {
                  int tmp_x =xi;
                  int x_return = x_start+xi;
                  data[to_data][y_return][x_return] = KERNEL(tmp[to_tmp], tmp_x, tmp_y);
                }
              }
            }
          }
          //////
          else if(yo == yo_split_size-1){
            int y_start = yo*BY-BT;
            int y_end = NY;
            if(xo == 0){
              int x_start = 0;
              int x_end = BX+BT;
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x =xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug
              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;
                for (yi = y_start+1+ti; yi < y_end-1; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1; xi < x_end-1-ti; xi++) {
                    int tmp_x =xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = BT; yi < BY+BT; yi++) {
                int tmp_y = yi;
                for (xi = 1; xi < BX; xi++) {
                  int tmp_x = xi;
                  data[to_data][yi][xi] = KERNEL(tmp[to_tmp], tmp_x, tmp_y);
                }
              }
            }
            ///
            else if(xo == xo_split_size-1){
              int x_start = xo*BX-BT;
              int x_end = NX;
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x =xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug
              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;
                for (yi = y_start+1+ti; yi < y_end-1; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1+ti; xi < x_end-1; xi++) {
                    int tmp_x =xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = BT; yi < BY+BT; yi++) {
                int tmp_y = yi;
                int y_return = y_start+yi;
                for (xi = BT; xi < BX+BT; xi++) {
                  int tmp_x = xi;
                  int x_return = x_start+xi;
                  data[to_data][y_return][x_return] = KERNEL(tmp[to_tmp], tmp_x, tmp_y);
                }
              }
            }
            ///
            else{
              int x_start = xo*BX-BT;
              int x_end = (xo+1)*BX+BT;
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x = xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug
              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;
                for (yi = y_start+1+ti; yi < y_end-1; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1+ti; xi < x_end-1-ti; xi++) {
                    int tmp_x = xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = BT; yi < BY+BT; yi++) {
                int tmp_y = yi;
                int y_return = y_start+yi;
                for (xi = BT; xi < BX+BT; xi++) {
                  int tmp_x =xi;
                  int x_return = x_start+xi;
                  data[to_data][y_return][x_return] = KERNEL(tmp[to_tmp], tmp_x, tmp_y);
                }
              }
            }
          }
          //////
          else{
            int y_start = yo*BY-BT;
            int y_end = (yo+1)*BY+BT;
            if(xo == 0){
              int x_start = 0;
              int x_end = BX+BT;
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x =xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug
              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;
                for (yi = y_start+1; yi < y_end-1-ti; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1; xi < x_end-1-ti; xi++) {
                    int tmp_x =xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = BT; yi < BY+BT; yi++) {
                int y_return = y_start+yi;
                for (xi = 1; xi < BX; xi++) {
                  int x_return = x_start+xi;
                  data[to_data][y_return][x_return] = KERNEL(tmp[to_tmp], xi, yi);
                }
              }
            }
            ///
            else if(xo == xo_split_size-1){
              int x_start = xo*BX-BT;
              int x_end = NX;
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x =xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug
              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;
                for (yi = y_start+1; yi < y_end-1-ti; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1+ti; xi < x_end-1; xi++) {
                    int tmp_x =xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = BT; yi < BY+BT; yi++) {
                int tmp_y = yi;
                int y_return = y_start+yi;
                for (xi = BT; xi < BX+BT; xi++) {
                  int tmp_x = xi;
                  int x_return = x_start+xi;
                  data[to_data][y_return][x_return] = KERNEL(tmp[to_tmp], tmp_x, tmp_y);
                }
              }
            }
            ///
            else{
              int x_start = xo*BX-BT;
              int x_end = (xo+1)*BX+BT;
              for (yi = y_start+1; yi < y_end-1; yi++) {
                int tmp_y = yi-y_start;
                for (xi = x_start+1; xi < x_end-1; xi++) {
                  int tmp_x = xi-x_start;
                  tmp[0][tmp_y][tmp_x] = KERNEL(data[from_data], xi, yi);
                }
              }
              to_tmp = 0; // Endo debug
              for(ti = 1; ti < BT-1; ti++){
                from_tmp=ti%2;
                to_tmp=(ti+1)%2;
                for (yi = y_start+1+ti; yi < y_end-1-ti; yi++) {
                  int tmp_y = yi-y_start;
                  for (xi = x_start+1+ti; xi < x_end-1-ti; xi++) {
                    int tmp_x = xi-x_start;
                    tmp[to_tmp][tmp_y][tmp_x] = KERNEL(tmp[from_tmp], tmp_x, tmp_y);
                  }
                }
              }
              for (yi = BT; yi < BY+BT; yi++) {
                int tmp_y = yi;
                int y_return = y_start+yi;
                for (xi = BT; xi < BX+BT; xi++) {
                  int tmp_x =xi;
                  int x_return = x_start+xi;
                  data[to_data][y_return][x_return] = KERNEL(tmp[to_tmp], tmp_x, tmp_y);
                }
              }
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

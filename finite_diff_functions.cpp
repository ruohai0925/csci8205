#include "finite_diff_functions.hpp"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>


#include <omp.h>

int nthreads;
double delta_h;
double delta_t;
uint64_t N_DIM;
#define BLOCKING_SIZE 16

/* Swapping the loop index doing j first then i*/
void laplacian_seq_ji(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
  int tid;
  uint64_t i,j;
  uint64_t N = u[0].size();
  double lap_u;
  for(j = 1; j < N-1; j++){
    for(i = 1; i < N-1; i++){
      lap_u = u[i][j+1]-4*u[i][j]+u[i][j-1]+u[i+1][j]+u[i-1][j];
      newu[i][j] = mu*(lap_u)/(delta_h*delta_h); 
    }  
  } 
}

/* Base line Laplacian sequential code*/
void laplacian_seq(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
  int tid;
  uint64_t i,j;
  uint64_t N = u[0].size();
  double lap_u;
  for(i = 1; i < N-1; i++){
    for(j = 1; j < N-1; j++){
      lap_u = u[i][j+1]-4*u[i][j]+u[i][j-1]+u[i+1][j]+u[i-1][j];
      newu[i][j] = mu*(lap_u)/(delta_h*delta_h); 
    }  
  } 
}

/* Laplacian with */
void laplacian_seq_2(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
  int tid;
  uint64_t i,j;
  uint64_t N = u[0].size();
  for(i = 1; i < N-1; i++){
    for(j = 1; j < N-1; j++){
      newu[i][j] = u[i+1][j];
    }
    for(j = 1; j < N-1; j++){
      newu[i][j] += u[i-1][j];
    }
    for(j = 1; j < N-1; j++){
      newu[i][j] += u[i][j+1]-4*u[i][j]+u[i][j-1];
      newu[i][j] = mu*(newu[i][j])/(delta_h*delta_h); 
    }  
  } 
}

/* Laplacian code with blocking for cache, blocking size is defined by BLOCKING_SIZE*/
void laplacian_seq_block(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
  int tid;
  uint64_t i,j,k,l,endk,endl;
  uint64_t N = u[0].size();
  double lap_u;
  for(i = 1; i < N-1; i += BLOCKING_SIZE){
    for(j = 1; j < N-1; j += BLOCKING_SIZE){
      endk = i + BLOCKING_SIZE;
      if(endk > N-1) endk=N-1;
      for(k = i; k < endk; k++){
        endl = j + BLOCKING_SIZE;
        if(endl > N-1) endl=N-1;
        for(l = j; l < endl; l++){
          lap_u = u[k][l+1]-4*u[k][l]+u[k][l-1]+u[k+1][l]+u[k-1][l];
          newu[k][l] = mu*(lap_u)/(delta_h*delta_h);
        }
      }
       
    }  
  } 
}

/* Laplacian code with blocking for cache rearranged in checkerboard style
    0  1  2  3  4 
    5  6  7  8  9
    10 11 12 13 14
    15 16 17 18 19
    20 21 22 23 24
    Numbers indicates block number
*/
void laplacian_seq_block_proto(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
  uint64_t i,j,k,l,endk,endl,tid,n;
  uint64_t N = u[0].size(),M;
  M = (N-2)/BLOCKING_SIZE+1;
  uint64_t MM = M*M;
  double lap_u;
  for(n = 0; n < MM; n++){
    i = (n%M)*BLOCKING_SIZE+1;
    j = (n/M)*BLOCKING_SIZE+1;
    endk = i + BLOCKING_SIZE;
    //printf("(%d,%d), M = %d, MM = %d, n = %d\n",i,j,M,MM,n);
    if(endk > N-1) endk=N-1;
    for(k = i; k < endk; k++){
      endl = j + BLOCKING_SIZE;
      if(endl > N-1) endl=N-1;
      for(l = j; l < endl; l++){
        lap_u = u[k][l+1]-4*u[k][l]+u[k][l-1]+u[k+1][l]+u[k-1][l];
        newu[k][l] = mu*(lap_u)/(delta_h*delta_h);
      }
    }
  }
}

/* OpenMP implementation of discrete laplacian*/
void laplacian_omp(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
  int tid;
  uint64_t i,j,Row_len,act_len;
  double lap_u;
  act_len = (N_DIM-1)/nthreads;
  #pragma omp parallel private(tid,i,j,lap_u,Row_len)
  {
    tid = omp_get_thread_num(); 
    if(tid != nthreads-1){
      Row_len = 1+(tid+1)*act_len; // How many rows each thread works on
    }
    else{
      Row_len = N_DIM;
    }
    //printf("Thread id = %d, [%d,%d] of [%d,%d]\n",tid,1+tid*act_len, Row_len,0,N_DIM);
    for(i = 1+tid*act_len; i < Row_len; i++){
      for(j = 1; j < N_DIM; j++){
        lap_u = u[i][j+1]-4*u[i][j]+u[i][j-1]+u[i+1][j]+u[i-1][j];
        newu[i][j] = (mu*lap_u)/(delta_h*delta_h); 
      }  
    }
  } 
}


/* Laplacian code with blocking for cache rearranged in rows style
    Example with 2 threads:
    0  1  2  3  4 
    5  6  7  8  9
    0  1  2  3  4
    5  6  7  8  9
    10 11 12 13 14
    Parallel Version, threads does each block in rotation.
*/
void laplacian_omp_blocking(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
  int tid;
  uint64_t i,j,k,l,Row_len,act_len,endl,endk;
  double lap_u;
  act_len = (N_DIM-1)/nthreads;
  #pragma omp parallel private(tid,i,j,k,l,endk,endl,lap_u,Row_len)
  {
    tid = omp_get_thread_num(); 
    if(tid != nthreads-1){
      Row_len = 1+(tid+1)*act_len; // How many rows each thread works on
    }
    else{
      Row_len = N_DIM;
    }
    //printf("Thread id = %d, [%d,%d] of [%d,%d]\n",tid,1+tid*act_len, Row_len,0,N_DIM);
    for(i = 1+tid*act_len; i < Row_len; i += BLOCKING_SIZE){
      for(j = 1; j < N_DIM; j += BLOCKING_SIZE){
        endk = i + BLOCKING_SIZE;
        if(endk > N_DIM) endk=N_DIM;
        for(k = i; k < endk; k++){
          endl = j + BLOCKING_SIZE;
          if(endl > N_DIM) endl=N_DIM;
          for(l = j; l < endl; l++){
            lap_u = u[k][l+1]-4*u[k][l]+u[k][l-1]+u[k+1][l]+u[k-1][l];
            newu[k][l] = mu*(lap_u)/(delta_h*delta_h);
          }
        }
      }  
    }
  } 
}


/* Laplacian code with blocking for cache rearranged in checkerboard style
    0  1  2  3  4 
    5  6  7  8  9
    10 11 12 13 14
    15 16 17 18 19
    20 21 22 23 24
    Parallel Version, threads does each block in rotation.
*/
void laplacian_omp_blocking_2(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
  int tid;
  uint64_t i,j,k,l,endk,endl,M = (N_DIM-2)/BLOCKING_SIZE+1,n;
  uint64_t MM = M*M;
  double lap_u;
  #pragma omp parallel private(tid,i,j,k,l,endk,endl,lap_u,n)
  {
    tid = omp_get_thread_num(); 
    for(n = tid; n < MM; n += nthreads){
        i = (n%M)*BLOCKING_SIZE+1;
        j = (n/M)*BLOCKING_SIZE+1;
        endk = i + BLOCKING_SIZE;
        if(endk > N_DIM) endk=N_DIM;
        for(k = i; k < endk; k++){
          endl = j + BLOCKING_SIZE;
          if(endl > N_DIM) endl=N_DIM;
          for(l = j; l < endl; l++){
            lap_u = u[k][l+1]-4*u[k][l]+u[k][l-1]+u[k+1][l]+u[k-1][l];
            newu[k][l] = mu*(lap_u)/(delta_h*delta_h);
          }
        }
      }
  } 
}

void construct_b(std::vector<std::vector<double> >&u,
                 std::vector<std::vector<double> >&newu,
                 std::vector<std::vector<double> >&v,
                 std::vector<std::vector<double> >&newv,
                 std::vector<std::vector<double> >&b){
    double ux,uy,vx,vy;
    uint64_t i,j,Row_len, act_len = (N_DIM-1)/nthreads;
    int tid;
    //start_core_measure(0);   
    #pragma omp parallel private(tid,i,j,Row_len,ux,uy,vx,vy)
    {
      tid = omp_get_thread_num();      
      if(tid != nthreads-1){
        Row_len = 1+(tid+1)*act_len; // How many rows each thread works on
      }
      else{
        Row_len = N_DIM;
      }
      //printf("tid = %d, with act_len = %d, and row_len = %d\n",tid,act_len,Row_len); 
      for(i = 1+tid*act_len; i < Row_len; i++){
        for(j = 1; j < N_DIM; j++){
          uy = (u[i][j+1]-u[i][j-1])/(delta_h*2);
          ux = (u[i+1][j]-u[i-1][j])/(delta_h*2);
          newu[i][j] = newu[i][j]-(u[i][j]*ux+v[i][j]*uy);
          vy = (v[i][j+1]-v[i][j-1])/(delta_h*2);
          vx = (v[i+1][j]-v[i-1][j])/(delta_h*2);
          newv[i][j] = newv[i][j]-(u[i][j]*vx+v[i][j]*vy);
          b[i][j] = -rho*delta_h*delta_h/4*(ux*ux+vy*vy+2*vx*uy - (ux+vy)/delta_t); 
      }  
     }   
    } 
}

/* Rearranged the delta_h*/
void construct_b_2(std::vector<std::vector<double> >&u,
                 std::vector<std::vector<double> >&newu,
                 std::vector<std::vector<double> >&v,
                 std::vector<std::vector<double> >&newv,
                 std::vector<std::vector<double> >&b){
    double ux,uy,vx,vy;
    uint64_t i,j,Row_len, act_len = (N_DIM-1)/nthreads;
    int tid;
    //start_core_measure(0);   
    #pragma omp parallel private(tid,i,j,Row_len,ux,uy,vx,vy)
    {
      tid = omp_get_thread_num();      
      if(tid != nthreads-1){
        Row_len = 1+(tid+1)*act_len; // How many rows each thread works on
      }
      else{
        Row_len = N_DIM;
      }
      //printf("tid = %d, with act_len = %d, and row_len = %d\n",tid,act_len,Row_len); 
      for(i = 1+tid*act_len; i < Row_len; i++){
        for(j = 1; j < N_DIM; j++){
          uy = (u[i][j+1]-u[i][j-1]);
          ux = (u[i+1][j]-u[i-1][j]);
          newu[i][j] = newu[i][j]-(u[i][j]*ux+v[i][j]*uy)/(delta_h*2);
          vy = (v[i][j+1]-v[i][j-1]);
          vx = (v[i+1][j]-v[i-1][j]);
          newv[i][j] = newv[i][j]-(u[i][j]*vx+v[i][j]*vy)/(delta_h*2);
          b[i][j] = -rho*(0.0625*(ux*ux+vy*vy+2*vx*uy) - delta_h*0.125*(ux+vy)/delta_t); 
      }  
     }   
    } 
}

/* With cache blocking row-wise*/
void construct_b_blocking(std::vector<std::vector<double> >&u,
                 std::vector<std::vector<double> >&newu,
                 std::vector<std::vector<double> >&v,
                 std::vector<std::vector<double> >&newv,
                 std::vector<std::vector<double> >&b){
    double ux,uy,vx,vy;
    uint64_t i,j,k,l,Row_len,endl,endk,act_len = (N_DIM-1)/nthreads;
    int tid;
    #pragma omp parallel private(tid,i,j,Row_len,ux,uy,vx,vy)
    {
      tid = omp_get_thread_num();      
      if(tid != nthreads-1){
        Row_len = 1+(tid+1)*act_len; // How many rows each thread works on
      }
      else{
        Row_len = N_DIM;
      }
      //printf("tid = %d, with act_len = %d, and row_len = %d\n",tid,act_len,Row_len); 
      for(i = 1+tid*act_len; i < Row_len; i += BLOCKING_SIZE){
        for(j = 1; j < N_DIM; j += BLOCKING_SIZE){
          endk = i + BLOCKING_SIZE;
          if(endk > N_DIM) endk=N_DIM;
          for(k = i; k < endk; k++){
            endl = j + BLOCKING_SIZE;
            if(endl > N_DIM) endl=N_DIM;
            for(l = j; l < endl; l++){
              uy = (u[k][l+1]-u[k][l-1]);
              ux = (u[k+1][l]-u[k-1][l]);
              newu[k][l] = newu[k][l]-(u[k][l]*ux+v[k][l]*uy)/(delta_h*2);
              vy = (v[k][l+1]-v[k][l-1]);
              vx = (v[k+1][l]-v[k-1][l]);
              newv[k][l] = newv[k][l]-(u[k][l]*vx+v[k][l]*vy)/(delta_h*2);
              b[k][l] = -rho*(0.0625*(ux*ux+vy*vy+2*vx*uy) - delta_h*0.125*(ux+vy)/delta_t); 
            }
          }
          
      }  
     }   
    } 
}

double construct_p(std::vector<std::vector<double> >&P,
                 std::vector<std::vector<double> >&newp,
                 std::vector<std::vector<double> >&b){
  uint64_t i,j,Row_len, act_len = (N_DIM-1)/nthreads, N = P.size();
  int tid;
  std::vector<double> t_diff;
  t_diff.resize(nthreads);
  for(i = 0; i < nthreads; i++){
    t_diff[i] = 0;
  }
  double diff;
  #pragma omp parallel private(tid,i,j,Row_len,diff)
  {
    tid = omp_get_thread_num();  
    diff = 0;    
    if(tid != nthreads-1){
      Row_len = 1+(tid+1)*act_len; // How many rows each thread works on
    }
    else{
      Row_len = N_DIM;
    }
    for(i = 1+(tid)*act_len; i < Row_len; i++){
      for(j = 1; j < N-1; j++){
        newp[i][j] = (P[i+1][j]+P[i-1][j]+P[i][j+1]+P[i][j-1])/4-b[i][j];
      } 
    }
  }
  for(i = 0; i < N; i++){
    newp[i][N-1] = 0; //y=1
  }
  for(i = 0; i < N; i++){
    newp[i][0] = newp[i][1]; // y=0
  }
  #pragma omp parallel private(tid,i,j,Row_len)
  {
    tid = omp_get_thread_num();    
    
    if(tid == 0){
      for(j = 0; j < N; j++){
        newp[0][j] = newp[1][j]; //x=0
      }
    }
    if(tid == nthreads-1){
      for(j = 0; j < N; j++){
        newp[N-1][j] = newp[N-2][j]; //x=1
      }
    }
  }
  act_len = (N_DIM-1)/nthreads;
  #pragma omp parallel private(tid,i,j,Row_len,diff)
  {
    tid = omp_get_thread_num();      
    if(tid != nthreads-1){
      Row_len = (tid+1)*act_len; // How many rows each thread works on
    }
    else{
      Row_len = N_DIM+1;
    }
    for(i = tid*act_len; i < Row_len; i++){
      for(j = 0; j < N; j++){
        diff = (P[i][j] - newp[i][j]);
        diff = (diff < 0) ? -diff : diff;
        if(diff > t_diff[tid]) t_diff[tid] = diff;
        P[i][j] = newp[i][j]; // copying
      } 
    }
  }
  diff = 0;
  for(i = 0; i < nthreads; i++){
    if(diff < t_diff[i]) diff = t_diff[i];
  }
  return diff;
}

double check_steady_state(std::vector<std::vector<double> >&Px,
                        std::vector<std::vector<double> >&Py,
                        std::vector<std::vector<double> >&newu,
                        std::vector<std::vector<double> >&newv){
  double steady_u, steady_v,steady,max_steady = 0;
  std::vector<double> steady_vals;
  steady_vals.resize(nthreads);
  double ux,uy,vx,vy;
  uint64_t N = Px.size();
  uint64_t i,j,Row_len, act_len = (N_DIM-1)/nthreads;
  int tid;
  //start_core_measure(0);   
  #pragma omp parallel private(tid,i,j,Row_len, steady_u, steady_v, steady)
  {
    steady = 0;
    tid = omp_get_thread_num();      
    if(tid != nthreads-1){
      Row_len = 1+(tid+1)*act_len; // How many rows each thread works on
    }
    else{
      Row_len = N_DIM;
    }
    for(i = 1+(tid)*act_len; i < Row_len; i++){
      for(j = 1; j < N-1; j++){
        steady_u = newu[i][j] - Px[i][j]/rho;
        steady_u = (steady_u < 0) ? -steady_u : steady_u;
        steady_v = newv[i][j] - Py[i][j]/rho;
        steady_v = (steady_v < 0) ? -steady_v : steady_v;
        if(steady < steady_u + steady_v) steady = steady_u + steady_v;
      } 
    }
    steady_vals[tid] = steady;
  }
  for(i = 0; i < nthreads; i++){
    if(max_steady < steady_vals[i]) max_steady = steady_vals[i];
  }
  return max_steady;
}

void print_vel(std::vector<std::vector<double> >&u, std::vector<std::vector<double> >&v, uint64_t N){
    for(int j = N-1; j >=0; j--){
    for(int i = 0; i < N; i++){
      printf("<%f,%f> ",u[i][j],v[i][j]);
       //printf("(%f)=%f ",b[i][j],p[i][j]);
    }
    printf("\n");
  } 
}

void print_vel_norm(std::vector<std::vector<double> >&u, std::vector<std::vector<double> >&v, uint64_t N){
    for(int j = N-1; j >=0; j--){
    for(int i = 0; i < N; i++){
      printf("%f ",sqrt(u[i][j]*u[i][j]+v[i][j]*v[i][j]));
       //printf("(%f)=%f ",b[i][j],p[i][j]);
    }
    printf("\n");
  } 
}

void print_value(std::vector<std::vector<double> >&u, uint64_t N){
    for(int j = N-1; j >=0; j--){
    for(int i = 0; i < N; i++){
      printf("%f ",u[i][j]);
       //printf("(%f)=%f ",b[i][j],p[i][j]);
    }
    printf("\n");
  } 
}

void print_value_list(double *u,uint64_t N){
    for(int j = N-1; j >=0; j--){
    for(int i = 0; i < N; i++){
      printf("%f ",u[i+j*N]);
       //printf("(%f)=%f ",b[i][j],p[i][j]);
    }
    printf("\n");
  } 
}
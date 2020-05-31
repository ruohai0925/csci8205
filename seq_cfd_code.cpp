
#include "seq_cfd_code.hpp"
#include "finite_diff_functions.hpp"

#include <math.h>

#include <stdio.h>
#include <stdlib.h>

#include <assert.h>
#include <time.h>


//#include "cuda_code.h"

#include <sys/sysinfo.h> 
#include <sys/time.h>

static void laplacian(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu){
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

static void convection(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu,
                std::vector<std::vector<double> >&v,std::vector<std::vector<double> >&newv){
  uint64_t i,j = 0;
  uint64_t N = u[0].size();
  double ux,uy,vx,vy;
  for(i = 1; i < N-1; i++){
    for(j = 1; j < N-1; j++){
      uy = (u[i][j+1]-u[i][j-1])/(delta_h*2);
      ux = (u[i+1][j]-u[i-1][j])/(delta_h*2);
      vy = (v[i][j+1]-v[i][j-1])/(delta_h*2);
      vx = (v[i+1][j]-v[i-1][j])/(delta_h*2);
      newu[i][j] = newu[i][j]-(u[i][j]*ux+v[i][j]*uy);
      newv[i][j] = newv[i][j]-(u[i][j]*vx+v[i][j]*vy);
    }
  }
}

static void build_up_b(std::vector<std::vector<double> > &b,  
                std::vector<std::vector<double> > &u, 
                std::vector<std::vector<double> > &v){
  uint64_t i,j;
  uint64_t N = u[0].size();
  double ux,uy,vx,vy;
  for(i = 1; i < N-1; i++){
    for(j = 1; j < N-1; j++){
      uy = (u[i][j+1]-u[i][j-1])/(delta_h*2);
      ux = (u[i+1][j]-u[i-1][j])/(delta_h*2);
      vy = (v[i][j+1]-v[i][j-1])/(delta_h*2);
      vx = (v[i+1][j]-v[i-1][j])/(delta_h*2);
      b[i][j] = rho*((ux+vy)/delta_t-(ux*ux+vy*vy+2*vx*uy));
    }
  }
}

static double pressure_poisson(std::vector<std::vector<double> > &p, 
                std::vector<std::vector<double> > &b,
                std::vector<std::vector<double> > &temp_p){
  uint64_t i,j;
  uint64_t N = p[0].size();
  double ux,uy,vx,vy;
  for(i = 1; i < N-1; i++){
    for(j = 1; j < N-1; j++){
      temp_p[i][j] = (p[i+1][j]+p[i-1][j]+p[i][j+1]+p[i][j-1])/4-delta_h*delta_h/4*b[i][j];
    }
  }
  for(i = 0; i < N; i++){
    temp_p[i][N-1] = 0;
  }
  for(i = 0; i < N; i++){
    temp_p[i][0] = temp_p[i][1];
  }
  for(j = 0; j < N; j++){
    temp_p[0][j] = temp_p[1][j];
  }
  for(j = 0; j < N; j++){
    temp_p[N-1][j] = temp_p[N-2][j];
  }
  double diff,max_diff=0;
  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      diff = (p[i][j] - temp_p[i][j]);
      diff = (diff < 0) ? -diff : diff;
      if(diff > max_diff) max_diff = diff;
      p[i][j] = temp_p[i][j];
    }
  }
  return max_diff;
}

bool verify_values(std::vector<std::vector<double> > &P, 
                std::vector<std::vector<double> > &U, 
                std::vector<std::vector<double> > &V, 
                uint64_t num_it){

  uint64_t N = U.size();
  bool veri = true;
  double px,py;
  uint64_t i,j,k;
  /* Allocate memory for u,v,p, and other stuff*/
  printf("==========Paramters==========\n");
  printf("delta_h = %.9f\n",delta_h);
  printf("delta_t = %.9f\n",delta_t);
  printf("mu = %.9f\n",mu);
  printf("rho = %.9f\n",rho);
  printf("=============================\n");
  std::vector< std::vector<double> > u;
  std::vector< std::vector<double> > v;
  std::vector< std::vector<double> > p;
  std::vector< std::vector<double> > b;
  std::vector< std::vector<double> > newu;
  std::vector< std::vector<double> > newv;
  std::vector< std::vector<double> > newp;

  u.resize(N);
  v.resize(N);
  p.resize(N);
  b.resize(N);
  newu.resize(N);
  newv.resize(N);
  newp.resize(N);    

  for(i = 0 ; i < N ; ++i){
    u[i].resize(N);
    v[i].resize(N);
    p[i].resize(N);
    b[i].resize(N);
    newu[i].resize(N);
    newv[i].resize(N);
    newp[i].resize(N); 
  }

  for(i = 0; i < N;i++){
    for(int j = 0; j < N; j++){
      u[i][j] = 0;
      v[i][j] = 0;
      if(j == N-1){
        u[i][j] = 1.0;
      }
     }
  }
  for(k = 0; k < num_it; k++){
    laplacian(u,newu);
    laplacian(v,newv);
    convection(u,newu,v,newv);

    build_up_b(b, u, v);
    double p_diff = 1;
    uint64_t l = 0;
    while(p_diff > delta_h/10.0){
      p_diff = pressure_poisson(p, b, newp);
      l++;
      if(l > 400) p_diff = 0;
    }

    for(i = 1; i < N-1; i++){
      for(j = 1; j < N-1; j++){
        px = (p[i+1][j]-p[i-1][j])/(2*delta_h);
        py = (p[i][j+1]-p[i][j-1])/(2*delta_h);
        u[i][j] = u[i][j]+delta_t*(newu[i][j]-px/rho);
        v[i][j] = v[i][j]+delta_t*(newv[i][j]-py/rho);
        assert(u[i][j] == u[i][j] || v[i][j] == v[i][j]);
      }
    }
  }
  double diff;

  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      diff = U[i][j] - u[i][j];
      diff = (diff < 0) ? -diff : diff;
      if(diff > 0.00001){
        veri = false;
        printf("U:: %.9f != %.9f (%d,%d)\n",U[i][j],u[i][j],i,j);
      }
      diff = V[i][j] - v[i][j];
      diff = (diff < 0) ? -diff : diff;
      if(diff > 0.00001){
        veri = false;
        printf("V:: %.9f != %.9f (%d,%d)\n",V[i][j],v[i][j],i,j);
      }
      diff = P[i][j] - p[i][j];
      diff = (diff < 0) ? -diff : diff;
      if(diff > 0.00001){
        veri = false;
        printf("P:: %.9f != %.9f (%d,%d)\n",P[i][j],p[i][j],i,j);
      }
    }

  }
  return veri;
}

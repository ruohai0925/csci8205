#include <math.h>

#include <mpi.h>
#include <omp.h>

#include "pcm_utils.hpp"
#include "seq_cfd_code.hpp" // Verify values with sequential version
#include "finite_diff_functions.hpp"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <assert.h>
#include <time.h>
#include <vector>
#include <fstream>
//#include "cuda_code.h"

#include <sys/sysinfo.h> 
#include <sys/time.h>



void get_node_info(int node_id){
  struct sysinfo myinfo; 
  unsigned long total_bytes; 
  sysinfo(&myinfo); 
  total_bytes = myinfo.mem_unit * myinfo.totalram; 
  int max_threads = omp_get_num_procs();
  printf("Node %d has %d threads\n",node_id,max_threads);
  printf("With Main Memory: %lu GB\n", total_bytes/1024/1024/1024); 
}

void all_node_info(int max_nodes){
	int node_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
	for(int i = 0 ; i < max_nodes; i++){
		if(node_id == i){
		  	get_node_info(node_id);
//		  	device_info();
		  	printf("======================================\n");
		  }
  		MPI_Barrier(MPI_COMM_WORLD);
	}
}
void laplacian_timing_tests();


int main(int argc, char *argv[]){
  //int cpu_list[2] = {0,1};
  //set_cpu(cpu_list,2);
  /*id = node number
    num_node = number of nodes
    provided = level of threading provided by MPI, we expect to have 3.*/
  int node_id,num_node, provided; 
  
  int ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided); /* Pass arguments to all nodes*/
  
  /* Get node id and node size*/
  ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_node);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
  
  if(node_id == 0){
  	/* Print the thread level provided by MPI library */
  	printf("Thread level provided by MPI library: %d\n",provided);

  }

  /* Can be used to find out node information */

  all_node_info(num_node);
  
  /* Find out which GPU devices have more than 6GB memory */
//  init_gpu_devices();


  /* Get thread info and set thread number based on command line argument*/
  int tid;
  nthreads = atoi(argv[1]);
  
  /* Problem specific values*/
  N_DIM = atoi(argv[2]);
  uint64_t N = N_DIM+1;
  delta_h = (SIDE_LEN*1.0)/(1.0*N_DIM);
  delta_t = delta_h*delta_h/4;
  double t = 0;

  /* set the number of threads each nodes will use, this is passed from the command line 
  tid = thread id */
  omp_set_num_threads(nthreads);
  uint64_t i,j,k,Row_len,act_len;

  /* Allocate memory for u,v,p, and other stuff*/

  std::vector< std::vector<double> > u;
  std::vector< std::vector<double> > v;
  std::vector< std::vector<double> > newu;
  std::vector< std::vector<double> > newv;
  std::vector< std::vector<double> > newp;
  std::vector< std::vector<double> > Px;
  std::vector< std::vector<double> > Py;
  std::vector< std::vector<double> > b;
  std::vector< std::vector<double> > P;
  u.resize(N);
  v.resize(N);
  b.resize(N);
  newu.resize(N);
  newv.resize(N);
  newp.resize(N);
  Px.resize(N);
  P.resize(N);
  Py.resize(N);     

  for(int i = 0 ; i < N ; ++i){
  	u[i].resize(N);
  	v[i].resize(N);
  	newu[i].resize(N);
  	newv[i].resize(N);
    Px[i].resize(N);
    Py[i].resize(N);  
  	P[i].resize(N); 
  	b[i].resize(N); 
    newp[i].resize(N);
  }
  for(int i = 0; i < N;i++){
    for(int j = 0; j < N; j++){
	    u[i][j] = 0;
      v[i][j] = 0;
      P[i][j] = 0;
      if(j == N-1){
        u[i][j] = 1.0;
      }
     }
  }

  printf("Done allocating memory for u,v,p, with h=%.9f and t=%.9f\n",delta_h,delta_t);

  double ux,vy,uy,vx,px,py;

  struct timeval  tv1, tv2;
  double res_time;
  uint64_t num_it = atoi(argv[3]),count = 0;// frame = atoi(argv[4]);
  for(k = 0; k < num_it; k++){
    /* Version 3 */
    
    /* Measure Cache Misses */
    gettimeofday(&tv1, NULL);



    laplacian_omp_blocking(u,newu);
    laplacian_omp_blocking(v,newv);

    construct_b_2(u, newu, v, newv, b);

    /* Solve the Pressure Poisson equation */
    act_len = (N_DIM-1)/nthreads;
  	uint64_t l = 0;
    double p_diff = 1;
    while(p_diff > delta_h/10.0){
      p_diff = construct_p(P,newp,b);
      //printf("Global Iteration %d with p_diff = %.10f and l = %d\n",k,p_diff,l);
      l++;
      if(l > 400) p_diff = 0;
    }
    printf("Pressure solve took %d iterations\n",l);

    #pragma omp parallel private(tid,i,j,Row_len,px,py) // Compute the pressure gradient
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
       // Gradient of p
        // px
          py = (P[i][j+1]-P[i][j-1])/(delta_h*2);
          //py = (p[i+(j+1)*N]-p[i+(j-1)*N])/(delta_h*2);
        // py
          px = (P[i+1][j]-P[i-1][j])/(delta_h*2);
          //px = (p[i+1+j*N]-p[i-1+j*N])/(delta_h*2);
          //Px[i][j] = px;
          //Py[i][j] = py;
        // update newu
          u[i][j] = u[i][j]+delta_t*(newu[i][j]-px/rho);
          v[i][j] = v[i][j]+delta_t*(newv[i][j]-py/rho);
          assert(u[i][j] == u[i][j] || v[i][j] == v[i][j]);
      }  
     }  
    }
    //double steady = check_steady_state(Px,Py,newu,newv);

   /*if(steady < 0.00005){
     printf("Steady steady solution reached at time t=%.9f\n",t);
     k = num_it+1;
   }*/
    //printf("velocity at %d\n",k);
    //print_vel(u,v,N);
    //printf("P\n");
    //print_value(P,N);
    //printf("gradient p\n");
    //print_vel(Px,Py,N);

    gettimeofday(&tv2, NULL);
    t += delta_t;
    res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
    double Num_op = 8*(N-2)*(N-2)*2+(N-2)*(N-2)*25+(N-2)*(N-2)*4*(l-1);
    printf ("Total time = %f seconds with %f Gflops\n",res_time,(Num_op/1000000000.0)/res_time);
    /*if( k % frame == 0){
      printf("Current time t = %.9f\n",t);
      std::ofstream myfile;
      myfile.open("./re_50000_550/velpre.csv."+std::to_string(count));
      myfile << "x coord, y coord, u, v, p, norm\n";
      for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
          myfile << std::fixed << std::setprecision(14) << i*delta_h << "," << j*delta_h << "," << u[i][j] << "," << v[i][j] << "," << P[i][j] << "," << sqrt(u[i][j]*u[i][j]+v[i][j]*v[i][j]) << std::endl;
          //myfile << std::fixed << std::setprecision(14) << i*delta_h << "," << j*delta_h << "," << sqrt(u[i][j]*u[i][j]+v[i][j]*v[i][j]) << std::endl;
        } 
      }
      myfile.close();
      count++;
    }*/
  }
  printf("Checking values with sequential code...\n");
  bool verify = verify_values(P, u, v, num_it);
  if(verify) printf("u,v,p, values passed verification\n");
  else printf("u,v,p, values failed verification\n");

  // output


  /* Cleanup */


  /* End of MPI codes */
  ierr = MPI_Finalize();
  return 0;
}



void laplacian_timing_tests(){
  std::vector< std::vector<double> > u;
  std::vector< std::vector<double> > newu;
  u.resize(N_DIM+1);

  newu.resize(N_DIM+1);
  struct timeval  tv1, tv2;
  double res_time;

  for(int i = 0 ; i < N_DIM+1 ; ++i){
    u[i].resize(N_DIM+1);
    newu[i].resize(N_DIM+1);
  }
  cache_overwrite();
  gettimeofday(&tv1, NULL);
  laplacian_seq_ji(u,newu);
  gettimeofday(&tv2, NULL);
  res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("Total time = %f seconds (laplacian_seq_ji)\n",res_time);

  cache_overwrite();
  //start_core_measure(0); 
  gettimeofday(&tv1, NULL);
  laplacian_seq(u,newu);
  gettimeofday(&tv2, NULL);
  res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("Total time = %f seconds (laplacian_seq)\n",res_time);
  //end_core_measure(0);

  cache_overwrite();
  //start_core_measure(0);
  gettimeofday(&tv1, NULL);
  laplacian_seq_block(u,newu);
  gettimeofday(&tv2, NULL);
  res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("Total time = %f seconds (laplacian_seq_block)\n",res_time);
  //end_core_measure(0);

  cache_overwrite();
  gettimeofday(&tv1, NULL);
  laplacian_seq_block_proto(u,newu);
  gettimeofday(&tv2, NULL);
  res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("Total time = %f seconds (laplacian_seq_block_proto)\n",res_time);

  cache_overwrite();
  gettimeofday(&tv1, NULL);
  laplacian_seq_2(u,newu);
  gettimeofday(&tv2, NULL);
  res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("Total time = %f seconds (laplacian_seq_2)\n",res_time);

  cache_overwrite();
  gettimeofday(&tv1, NULL);
  laplacian_omp(u,newu);
  gettimeofday(&tv2, NULL);
  res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("Total time = %f seconds (laplacian_omp)\n",res_time);

  cache_overwrite();
  gettimeofday(&tv1, NULL);
  laplacian_omp_blocking(u,newu);
  gettimeofday(&tv2, NULL);
  res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("Total time = %f seconds (laplacian_omp_blocking)\n",res_time);

  cache_overwrite();
  gettimeofday(&tv1, NULL);
  laplacian_omp_blocking_2(u,newu);
  gettimeofday(&tv2, NULL);
  res_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("Total time = %f seconds (laplacian_omp_blocking_2)\n",res_time);
}
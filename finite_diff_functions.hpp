#ifndef FINITE_DIFF_FUNCTION_H
#define FINITE_DIFF_FUNCTION_H

#include <vector>
#include <stdint.h>

//#define N_DIM             19999 /*size in x dimension*/
#define SIDE_LEN          1

//#define BLOCK_SIZE        1
#define rho               1.0f
//#define mu                0.0003125f
#define mu                0.001f
//#define TIMESTEP          0.0001f

extern int nthreads;
extern double delta_h;
extern double delta_t;
extern uint64_t N_DIM;
extern uint64_t BLOCKING_SIZE;

void laplacian_seq_ji(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu);
void laplacian_seq(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu);
void laplacian_seq_2(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu);
void laplacian_seq_block(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu);
void laplacian_seq_block_proto(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu);
void laplacian_omp(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu);
void laplacian_omp_blocking(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu);
void laplacian_omp_blocking_2(std::vector<std::vector<double> >&u,std::vector<std::vector<double> >&newu);

void construct_b(std::vector<std::vector<double> >&u,
                 std::vector<std::vector<double> >&newu,
                 std::vector<std::vector<double> >&v,
                 std::vector<std::vector<double> >&newv,
                 std::vector<std::vector<double> >&b);

void construct_b_2(std::vector<std::vector<double> >&u,
                 std::vector<std::vector<double> >&newu,
                 std::vector<std::vector<double> >&v,
                 std::vector<std::vector<double> >&newv,
                 std::vector<std::vector<double> >&b);

void construct_b_blocking(std::vector<std::vector<double> >&u,
                 std::vector<std::vector<double> >&newu,
                 std::vector<std::vector<double> >&v,
                 std::vector<std::vector<double> >&newv,
                 std::vector<std::vector<double> >&b);

double construct_p(std::vector<std::vector<double> >&P,
                 std::vector<std::vector<double> >&newp,
                 std::vector<std::vector<double> >&b);

double check_steady_state(std::vector<std::vector<double> >&Px,
                        std::vector<std::vector<double> >&Py,
                        std::vector<std::vector<double> >&newu,
                        std::vector<std::vector<double> >&newv);

void print_vel(std::vector<std::vector<double> >&u, std::vector<std::vector<double> >&v, uint64_t N);

void print_vel_norm(std::vector<std::vector<double> >&u, std::vector<std::vector<double> >&v, uint64_t N);

void print_value(std::vector<std::vector<double> >&u, uint64_t N);

void print_value_list(double *u,uint64_t N);

#endif
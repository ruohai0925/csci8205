#ifndef SEQ_CFD_CODE_H
#define SEQ_CFD_CODE_H

#include <vector>
#include <stdint.h>


bool verify_values(std::vector<std::vector<double> > &P, 
                std::vector<std::vector<double> > &U, 
                std::vector<std::vector<double> > &V, 
                uint64_t num_it);

#endif
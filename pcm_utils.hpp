#ifndef PCM_UTIL_H
#define PCM_UTIL_H

#include "./pcm/cpucounters.h"

void set_cpu(int *cpu_list, int num_cpu);

void cache_overwrite();
void start_core_measure(int core_num);
void end_core_measure(int core_num);

#endif

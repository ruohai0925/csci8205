#include "pcm_utils.hpp"
#include <iostream>
#include <stdint.h>


CoreCounterState before;
CoreCounterState after;
PCM *m;

void set_cpu(int *cpu_list, int num_cpu){
    cpu_set_t set;
    CPU_ZERO(&set);        // clear cpu mask
    for(int i = 0; i < num_cpu; i++){
      CPU_SET(cpu_list[i], &set);    
    }
    sched_setaffinity(0, sizeof(cpu_set_t), &set);
}

void cache_overwrite(){
  std::vector<double> temp;
  temp.resize(1000000000);
  uint64_t j = 0;
  for(uint64_t i = 0; i < 1000000; i++){
    temp[i] = i;
    j += i;
  }
}

void start_core_measure(int core_num){
    if(m != NULL) return;
    m = PCM::getInstance();
    PCM::ErrorCode returnResult = m->program(PCM::DEFAULT_EVENTS, NULL);
    if (returnResult != PCM::Success){
        std::cerr << "Intel's PCM couldn't start" << std::endl;
        exit(1);
    }
    CoreCounterState before = getCoreCounterState(core_num);
}

void end_core_measure(int core_num){
    if(m == NULL) return;
    CoreCounterState after = getCoreCounterState(core_num);
    //SystemCounterState after = getSystemCounterState();
    std::cout << "===== Measurements =====\n";
    std::cout << "Instructions per clock: " << getIPC(before, after) << std::endl;
    std::cout << "Cycles per op: " << getCycles(before, after) << std::endl;
    std::cout << "------------------------\n";
    std::cout << "L1 Misses:     " << getL2CacheMisses(before, after) + getL2CacheHits(before, after)<< std::endl;
    std::cout << "------------------------\n";
    std::cout << "L2 Misses:     " << getL2CacheMisses(before, after) << std::endl;
    std::cout << "L2 Hits:       " << getL2CacheHits(before, after) << std::endl; 
    std::cout << "L2 hit ratio:  " << getL2CacheHitRatio(before, after) << std::endl;
    std::cout << "------------------------\n";
    std::cout << "L3 Misses:     " << getL3CacheMisses(before,after) << std::endl;
    std::cout << "L3 Hits:       " << getL3CacheHits(before, after) << std::endl;
    std::cout << "L3 hit ratio:  " << getL3CacheHitRatio(before, after) << std::endl;
    std::cout << "------------------------\n";
    m->cleanup();
}

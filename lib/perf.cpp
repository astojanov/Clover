#include <cstdint>
#include <cstddef>
#include <cstring>
#include <sys/types.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include "perf.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__) || defined(linux) || defined(__linux)
#include <sched.h>
#include <unistd.h>
#endif

using namespace std;

#ifndef WIN32
    #if defined(__GNUC__) || defined(__linux__)
        #define VOLATILE __volatile__
        #define ASM __asm__
    #else
        #define ASM asm
        #define VOLATILE
    #endif
#endif

#ifndef WIN32
    #define myInt64 unsigned long long
    #define INT32 unsigned int
#else
    #define myInt64 signed __int64
	#define INT32 unsigned __int32
#endif


#if defined(WIN32) || defined(_MSC_VER)
typedef union
	{
	    myInt64 int64;
        struct {
            INT32 lo;
            INT32 hi;
        } int32;
	} tsc_counter_t;

	#define RDTSC(cpu_c)   \
	{       __asm rdtsc    \
			__asm mov (cpu_c).int32.lo,eax  \
			__asm mov (cpu_c).int32.hi,edx  \
	}

	#define CPUID() \
	{ \
		__asm mov eax, 0 \
		__asm cpuid \
	}
#else
typedef union {
    myInt64 int64;
    struct {
        INT32 lo;
        INT32 hi;
    } int32;
} tsc_counter_t;

#define RDTSC(cpu_c) ASM VOLATILE ("rdtsc" : "=a" ((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))
#define CPUID()      ASM VOLATILE ("cpuid" : : "a" (0) : "bx", "cx", "dx" )
#endif

struct timeval failback_mode_tp_start;
struct timeval failback_mode_tp_stop;
tsc_counter_t  failback_mode_tsc_counter_start;
tsc_counter_t  failback_mode_tsc_counter_stop;

unsigned int num_cores = 0;
int * cpu_cores = NULL;

void init_cpu_cores ()
{

#if defined(__APPLE__)

    int count;
    size_t count_len = sizeof(count);
    sysctlbyname("hw.logicalcpu", &count, &count_len, NULL, 0);
    num_cores = (unsigned int) count;
    cpu_cores = new int[num_cores];

    for (int i = 0; i < num_cores; i += 1) {
        cpu_cores[i] = 1;
    }

#elif defined(__linux__) || defined(linux) || defined(__linux)

    FILE * f_presentcpus = fopen("/sys/devices/system/cpu/present", "r");
    if (!f_presentcpus) {
        std::cerr << "Can not open /sys/devices/system/cpu/present file." << std::endl;
        exit(1);
    }

    char buffer[1024];
    if(NULL == fgets(buffer, 1024, f_presentcpus)) {
        std::cerr << "Can not read /sys/devices/system/cpu/present." << std::endl;
        exit(1);
    }
    fclose(f_presentcpus);


    sscanf(buffer, "0-%d", &num_cores);
    if (num_cores == 0) {
        sscanf(buffer, "%d", &num_cores);
    }

    if (num_cores == 0) {
        std::cerr << "Can not read number of present cores" << std::endl;
        exit(1);
    } else {
        num_cores += 1;
        cpu_cores = new int[num_cores];
        memset(cpu_cores, 0, sizeof(int) * num_cores);
    }

    FILE * f_cpuinfo = fopen("/proc/cpuinfo", "r");
    if (!f_cpuinfo) {
        std::cerr << "Can not open /proc/cpuinfo file." << std::endl;
        exit(1);
    }

    while (0 != fgets(buffer, 1024, f_cpuinfo)) {
        if (strncmp(buffer, "processor", sizeof("processor") - 1) == 0) {
            int core_id;
            sscanf(buffer, "processor\t: %d", &core_id);
            cpu_cores[core_id] = 1;
        }
    }
    fclose(f_cpuinfo);

#endif

    for (int i = 0; i < num_cores; i += 1) {
        cout << "Core " << i << "\t: ";
        if (cpu_cores[i]) {
            cout << "online" << endl;
        } else {
            cout << "offline" << endl;
        }
    }
}




int get_highest_core ()
{
    int highest_core = 0;
    for (int i = 0; i < num_cores; i += 1) {
        if (cpu_cores[i]) {
            highest_core = i;
        }
    }
    cout << "Scheduling on core: " << highest_core << endl << endl;
    return highest_core;
}



void perf_init ()
{
    init_cpu_cores ();

//#if defined(__linux__) || defined(linux) || defined(__linux)
//    int cpu = get_highest_core ();
//    cpu_set_t cpu_set;
//    CPU_ZERO(&cpu_set);
//    CPU_SET(cpu, &cpu_set);
//    sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpu_set);
//#endif
}


void cycles_count_start ()
{
    gettimeofday(&failback_mode_tp_start, NULL);
    CPUID();
    RDTSC(failback_mode_tsc_counter_start);
}

benchmark_t cycles_count_stop () {
    RDTSC(failback_mode_tsc_counter_stop);
    CPUID();
    gettimeofday(&failback_mode_tp_stop, NULL);
    double sec_dff  = (double) (failback_mode_tp_stop.tv_sec  - failback_mode_tp_start.tv_sec);
    double usec_dff = (double) (failback_mode_tp_stop.tv_usec - failback_mode_tp_start.tv_usec);
    benchmark_t result;
    result.time = sec_dff + usec_dff * 1.e-6;
    result.cycles = (uint64_t)(failback_mode_tsc_counter_stop.int64 - failback_mode_tsc_counter_start.int64);
    return result;
}

bool cmp_benchmark_t(benchmark_t lhs, benchmark_t rhs) {
    return lhs.cycles < rhs.cycles;
}

void perf_done ()
{
    delete [] cpu_cores;
}

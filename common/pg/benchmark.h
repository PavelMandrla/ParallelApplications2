////////////////////////////////////////////////////////////////////////////////////////////////////
// file: benchmark.h
// version: 1.0
// author: Petr Gajdos
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <chrono> 
#include <iomanip> 
#include <iostream> 
#include <string> 
#include <algorithm>
#include <cudaDefs.h>

namespace cpubenchmark
{
	//typedef void(*timeableFn)(void);
	using  timeableFn = void(*)(void);

	inline double singleRun(timeableFn fn)
	{
		unsigned long i, count = 1;
		double timePer = 0;
		for (count = 1; count != 0; count *= 2) 
		{
			auto start = std::chrono::high_resolution_clock::now();
			for (i = 0; i<count; i++) 
			{
				fn();
			}
			auto end = std::chrono::high_resolution_clock::now();
			double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1.0e-9;
			const double time_granularity = 1.0e-3;
			timePer = elapsed / count;
			if (elapsed>time_granularity) /* That took long enough */
			{
				return timePer;
			}
		}
		/* woa-- if we got here, "count" reached integer wraparound before the timer ran long enough.  Return the last known time */
		return timePer;
	}

	inline double run(timeableFn fn, const unsigned int count)
	{
		double *times = new double[count];
		
		//Init run, the time will be not taken into account
		singleRun(fn);
		for (unsigned int t = 0; t<count; t++)
		{
			times[t] = singleRun(fn);
		}
		
		std::sort(&times[0], &times[count]);
		
		double rv = times[count >> 1];
		delete [] times;
		return rv;
	}

	// Print the time taken for some action, in our standard format.
	inline void print_ns(const std::string &what, double seconds) 
	{
		std::cout << what << ":\t" << std::setprecision(3) << seconds*1.0e9 << "\tns/" << std::endl;
	}

	// Time a function's execution, and print the time out in nanoseconds.
	inline void print_time(const std::string &fnName, timeableFn fn, const unsigned int count)
	{
		print_ns(fnName, run(fn, count));
	}
}

namespace gpubenchmark
{
	template <typename F>
	double run(F fn, const unsigned int count)
	{
		float* times = new float[count];

		cudaEvent_t startEvent, stopEvent;
		float elapsedTime;
		checkCudaErrors(cudaEventCreate(&startEvent));
		checkCudaErrors(cudaEventCreate(&stopEvent));

		//Init run, the time will be not taken into account
		fn();
		for (unsigned int t = 0; t < count; t++)
		{
			checkCudaErrors(cudaEventRecord(startEvent, 0));
			fn();
			checkCudaErrors(cudaEventRecord(stopEvent, 0));
			checkCudaErrors(cudaEventSynchronize(stopEvent));
			checkCudaErrors(cudaEventElapsedTime(&times[t], startEvent, stopEvent));
		}

		std::sort(&times[0], &times[count]);

		checkCudaErrors(cudaEventDestroy(startEvent));
		checkCudaErrors(cudaEventDestroy(stopEvent));

		double rv = times[count >> 1];
		delete[] times;
		return rv;
	}

	// Print the time taken for some action, in our standard format.
	inline void print_ns(const std::string& what, double ms)
	{
		std::cout << what << ":\t" << std::setprecision(3) << ms << "\tms/" << std::endl;
	}

	// Time a function's execution, and print the time out in milliseconds.
	template <typename F>
	void print_time(const std::string& fnName, F fn, const unsigned int count)
	{
		print_ns(fnName, run(fn, count));
	}
}

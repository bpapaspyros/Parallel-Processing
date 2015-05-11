#ifndef BTIMER_HPP
#define BTIMER_HPP

#include <omp.h>
#include <iostream>

class bTimer {
	public:
		bTimer() {
			startTimer = 0;
			timeDif = 0;

			sum = 0;
		};

		void startT() {
			startTimer = omp_get_wtime();
		};
		
		void stopT() {
			timeDif = omp_get_wtime()-startTimer;

			sum += timeDif;
		};

		void resetSum() {
			sum = 0;
		};

		void stopNprint() {
			stopT();

			std::cout << "Time: " << timeDif << std::endl;
		};

		void printSum() {
			std::cout << "Time sum: " << sum << std::endl;
		}

		double getTimeDif() {
			return timeDif;
		};

		double getStartT() {
			return startTimer;
		};

		double getSum() {
			return sum;
		};

	private:
		double startTimer;
		double timeDif;

		double sum;
};


#endif
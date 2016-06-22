### Parallel Processing assignment (4th semester course in CEID)

The purpose of this assignment was to implement/modify the given code to match the OpenMP standards as well as to use the Intel SIMD instructions to achieve better performance during the simulation. Furthermore, we attempted to minimize cache misses and maximize performance by using Score-P and other techniques.

### Compilation

`make`

### Execution

`./particles <framenum> <.fluid input file> [.fluid output file]`

### Authors/Copyright

I do not own the original code and algorithm. In this repository you will find the modified algorithm that runs with OpenMP and SIMDs. All modifications were made for educational purposes only. `Code originally written by Richard O. Lee and Christian Bienia`. More references to the original authors can be found in the comments at the top of each source file.

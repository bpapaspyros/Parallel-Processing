TARGET := particles
OBJS   := particles.o cellpool.o

CXX := g++
# CXX := scorep g++

# CXXFLAGS := -O3
# CXXFLAGS := -O3 -fopenmp
# CXXFLAGS := -O3 -fopenmp -msse2
CXXFLAGS := -fopenmp -msse2
# CXXFLAGS := -fopenmp -msse2
#CXXFLAGS := -g3
CXXFLAGS := $(CXXFLAGS) -Wall -W -Wmissing-declarations -Wredundant-decls -Wdisabled-optimization -Wextra
CXXFLAGS := $(CXXFLAGS) -Winline -Wpointer-arith -Wsign-compare -Wendif-labels

# To enable visualization comment out the following lines (don't do this for benchmarking)
# OBJS     += view.o
# CXXFLAGS += -DENABLE_VISUALIZATION
# LIBS     += -lGLU -lGL -lglut

all: particles compare

particles: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -c $<

compare: compare.cpp
	rm -rf compare
	$(CXX) compare.cpp -o compare
clean:
	rm -rf $(TARGET) $(OBJS) compare


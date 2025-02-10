CXX=g++
CC=gcc
GBENCHFLAGS=-D__GBENCH__ -lbenchmark_main -std=c++17 -I/usr/local/include -L/usr/local/lib -lbenchmark
GTESTFLAGS=-D__GTEST__ -DGTEST_HAS_PTHREAD=1 -lgtest_main -lgtest
OPTFLAGS=-O2
all : bench test

bench : main.cpp $(OBJS)
	$(CXX) -o bench main.cpp $(OPTFLAGS) $(GBENCHFLAGS)

test : main.cpp $(OBJS)
	$(CXX) -o test main.cpp $(OPTFLAGS) $(GTESTFLAGS)

clean :
	rm -rf bench test 

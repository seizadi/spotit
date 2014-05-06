INCLUDES = $(shell pkg-config --cflags opencv)
LIBS = $(shell pkg-config --libs opencv)
CXX = clang++
CXXFLAGS = -g -Wall -std=c++11 $(INCLUDES)
LDFLAGS = $(LIBS)

.PHONY: run

run: spot-it
	./spot-it ../spot-it-small.jpg

spot-it: main.cpp SpotIt.cpp
	$(LINK.cc) -o $@ $^

clean:
	rm -rf spot-it
	rm -rf spot-it.dSYM

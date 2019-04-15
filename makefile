
CXX = g++


CXXFLAGS = `upcxx-meta PPFLAGS` `upcxx-meta LDFLAGS`
LDFLAGS  = `upcxx-meta LIBFLAGS`

all: laplace_upc

laplace_upc: laplace_upcpp.cpp 
		$(CXX) laplace_upcpp.cpp -o laplace_upcpp $(CXXFLAGS) $(LDFLAGS)

clean:
		rm laplace_upcpp

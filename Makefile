CC=gcc
CXX=g++-4.6
CFLAGS=
#NVCCFLAGS=-Xcompiler "-fdump-tree-nrv"
NVCC=nvcc -arch=sm_21 -w

CUDA_ROOT=/usr/local/cuda

EXECUTABLES=
EXAMPLE_PROGRAM=benchmark example1 example2
OBJ=obj/device_matrix.o
 
.PHONY: debug all o3 ctags
all: libs $(EXECUTABLES) $(EXAMPLE_PROGRAM) ctags

o3: CFLAGS+=-O3
o3: all
debug: CFLAGS+=-g -DDEBUG
debug: all

libs: $(OBJ) lib/libcumatrix.a

lib/libcumatrix.a: $(OBJ)
	ar rcs $@ $<
	ranlib $@

vpath %.h include/
vpath %.cpp src/
vpath %.cu src/

INCLUDE= -I include/

LIBRARY= -lcuda -lcublas -lcudart
LIBRARY_PATH=-L$(CUDA_ROOT)/lib64/
CUDA_INCLUDE=$(INCLUDE) \
	     -isystem $(CUDA_ROOT)/samples/common/inc/ \
	     -isystem $(CUDA_ROOT)/include

CPPFLAGS= -std=c++0x $(CFLAGS) $(INCLUDE)

benchmark: $(OBJ) benchmark.cpp
	$(CXX) $(CFLAGS) $(CUDA_INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)
example1: $(OBJ) example1.cpp
	$(CXX) $(CFLAGS) $(CUDA_INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)
example2: $(OBJ) example2.cu
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(CUDA_INCLUDE) -o $@ $^ $(LIBRARY_PATH) $(LIBRARY)
# +==============================+
# +===== Other Phony Target =====+
# +==============================+
obj/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -o $@ -c $<

obj/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(CUDA_INCLUDE) -o $@ -c $<

obj/%.d: %.cpp
	@$(CXX) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@;\
	rm -f $@.$$$$

-include $(addprefix obj/,$(subst .cpp,.d,$(SOURCES)))

.PHONY: ctags
ctags:
ifneq ($(shell which ctags),)
	@ctags -R *
endif

clean:
	rm -rf $(EXECUTABLES) $(EXAMPLE_PROGRAM) obj/* lib/*.a

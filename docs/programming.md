Programming Physis
==================

The Physis DSL extends the standard C with several new data types and
intrinsics for stencil computations. The user is required to
use the extensions to express stencil-based applications, which are
then translated to actual implementation code by the Physis translator.
Note that our DSL currently supports the dimensionality
of up to three; it can be easily extended to support higher
dimension.

Runtime Initialization and Finalization
---------------------------------------

The user program must first initialize the Physis runtime environment
before any use of Physis extensions by the following intrinsic.

    void PSInit(int *argc, char ***argv,
       int num_dimensions, size_t max_dimx,
       size_t max_dimy, size_t max_dimz)

The first two parameters are assumed to be the command-line argument
number and pointers, as the `MPI_Init` function in MPI.
The additional parameter specifies the property of grids to be created
subsequently on the runtime. The `num_dimensions` parameter
specifies the maximum number of dimensions and 
the rest of parameters specifies the maximum size of each
dimension. Note that the number of additional parameters must be the
same as the number of dimensions.

Similarly, the Physis runtime can be destroyed by:

    void PSFinalize()

Using Multidimensional Grids
----------------------------

### Grid Data Types
Physis supports multidimensional Cartesian grids of floating point
values (either float or double). We currently do not
support grids of structs; they can be represented by using multiple
separate grids of floats or doubles.
To represent multidimensional grids, we introduce several new data
types named based on its dimensionality and element type, e.g.,
`PSGrid3DFloat` for 3-D grids of float values and `PSGrid2DDouble` for
2-D grids of double values.  
The type does not expose its
internal structure, but rather an opaque handle to actual
implementation, which may differ depending on translation targets.

Since many of the Physis intrinsics are overloaded with respect to the
grid types, we simply use `PSGrid` to specify different grid types
unless it is ambiguous.

### Creating and Deleting Grids

Grids of type `PSGridFloat3D` can be created with intrinsic 
  `PSGridFloat3DNew`. More specifically, the intrinsic is defined as
follows:

    PSGrid3DFloat PSGrid3DFloatNew(
        size_t dimx, size_t dimy, size_t dimz,
        enum PS_GRID_ATTRIBUTE attr)

The first three parameters specify the size of each of the three
dimensions. Similarly, intrinsics for creating double-type grids and 1-D
and 2-D grids are provided. The size of each dimension of grids can be
retrieved by intrinsic `PSGridDim`.

Parameter `attr` is an optional parameter to specify a set of
attributes. Currently the only available attribute is
`PS_GRID_CIRCULAR`, which designates that the grid to be created
allows circular boundary accesses.  


As in C, Physis does not automatically free heap-allocated
memory. Grids created by the above intrinsics can be manually deleted
by intrinsic defined as:

    void PSGridFree(PSGrid g)
    
### Grid Reads and Writes

Grids can be accessed both in bulk and point-wise ways. Bulk reads and
writes are:

    void PSGridCopyin(PSGrid g, const void *src)
    void PSGridCopyout(PSGrid g, void *dst)

`PSGridCopyin` copies the linear memory pointed by the second parameter
into the given grid, while `PSGridCopyout` copies the grid element values
into the memory pointed by the second parameter. The size of data copy is
determined by the element type and size of the given grid. Physis
assumes column-major order storage of multidimensional grids.

Each point of grids can be accessed using the following three intrinsics:

    // For 3-dimensional type-T grids
    T PSGridGet(PSGrid g,
                size_t i, size_t j, size_t k)
    void PSGridSet(PSGrid g,
                   size_t i, size_t j, size_t k, T v)
    void PSGridEmit(PSGrid g, T v)

The set of `size\_t` parameters specify the indices of a point
within the given grid, so  the number of index parameters depend on
the dimensionality of the grid (e.g., three for 3-D grids). The return
type of `PSGridGet` and the `v` parameter of `PSGridSet`
and `PSGridEmit` have the same type as the element type of the
grid, which is either `float` or `double`.

`PSGridGet` returns the value of the specified point, while
  `PSGridSet` writes a new value to the specified point. `PSGridEmit`
performs similarly to `PSGridSet`, but does not accept the index
parameters, and is solely used in stencil functions as described below.

Writing Stencils
----------------

### Stencil Functions

Stencils in Physis are expressed as *stencil functions*, which are
standard C functions with several restrictions. Stencil functions
represent a scalar computation of each point in grids. At runtime,
stencil functions may be executed sequentially or in
parallel. The following code illustrates a 7-point stencil function
for 3-D grids.

    void diffusion(const int x, const int y,
             PSGrid3DFloat g1, PSGrid3DFloat g2, float t) {
      float v = PSGridGet(g1,x,y,z)
          + PSGridGet(g1,x+1,y,z) + PSGridGet(g1,x-1,y,z)
          + PSGridGet(g1,x,y+1,z) + PSGridGet(g1,x,y-1,z)
          + PSGridGet(g1,x,y,z+1) + PSGridGet(g1,x,y,z-1);
      PSGridEmit(g2, v / 7.0 * t);
      return;
    }

The restrictions of stencil functions are as follows. First, the
function parameters must begin with `const int` parameters, which
represent the coordinate of the stencil point where this function is
applied, followed by any number of additional parameters, including
grids and other scalar values. Non-scalar parameters are not allowed in
stencil functions. The return type of stencil functions must be `void`.  

Second, calls within stencil functions must be either 1) calls to
intrinsics `PSGridGet`, `PSGridEmit`, or `PSGridDim`, 2)
calls to builtin math functions such as `sqrt` and `sin`, or
3) calls to other stencil functions.
The available math functions depend on a particular target platform,
since we simply redirect such calls to 
platform-native builtin functions; however, the user can mostly assume
availability of the standard libc math functions since they are
supported in CUDA. We expect this would be the case in other
accelerators than NVIDIA GPUs.
Other calls are assumed to be stencil functions too and are subject of
the same set of restrictions, except for the function parameter and
return type requirement.

Third, the stencil index arguments of `PSStencilGet` must match
the pattern of `x + c`, where `x` must be one of the index
parameters of the stencil function and `c` be an integral
immediate value. For example, `PSGridGet(g1, x, y, z)` in
the above example stencil is accepted by our translator, but 
nor is `PSGridGet(g1, x + t, y, z)`, where `t` is not an immediate value
but a given parameter. Furthermore, the order of index parameters
appearing in `PSStencilGet` must match the order of the parameters of
the stencil function. For example, `PSGridGet(g1, z, y, x)` is not
legal in Physis. This requirement allows us to assume that data
dependencies between stencil points can be resolved by neighbor data
exchanges.

Fourth, in stencil functions, aliases of grid variables must be
unambiguously analyzable. Since our current translator supports only
a very simple alias analysis, each grid variable must follow the form
of static single assignments in the function. Also, taking the address
of a grid variable is not allowed in stencil functions.

Finally, a stencil function may be executed in parallel for grid points
with an arbitrary order, so the programmer must not assume any
read-after-write dependency among different stencil points within a
function. Such dependency can only be enforced between different
invocations of stencil functions.

These restrictions are to enforce regular neighbor data accesses
patterns in stencil functions, and to allow for static 
generation of efficient parallel code. Some of them could be relaxed
and complimented by runtime analysis and code generation. For example,
we could allow for arbitrary index arguments  in `PSGridGet`, but
accessing arbitrary points could be a significant performance bottleneck.
Since our current framework prioritizes efficiency of generated code
over flexibility, the translator does not accept code that violates
the above restrictions. Other C constructs such as branches and loops
are accepted.

### Applying Stencils to Grids

Stencil functions can be applied to grids by using two declarative
intrinsics: `PSStencilMap` and `PSStencilRun`.
The following code illustrates how these intrinsics can be used to
invoke the diffusion stencil on 3-D grids.

    PSInit(&argc, &argv, 3, NX, NY, NZ);
    PSGrid3DFloat g1 = PSGrid3DFloatNew(NX, NY, NZ);
    PSGrid3DFloat g2 = PSGrid3DFloatNew(NX, NY, NZ);
    // initial_data is a pointer to input data 
    PSGridCopyin(g1, initial_data);
    PSDomain3D d = PSDomain3DNew(0, NX, 0, NY, 0, NZ);
    PSStencilRun(PSStencilMap(diffusion, d, g1, g2, 0.5),
                 PSStencilMap(diffusion, d, g2, g1, 0.5),
                 10);
    // result is a pointer to hold result  data 
    PSCopyout(g1, result);
    PSFinalize();


`PSStencilMap` creates an 
object of `PSStencil`, which encapsulates a given stencil
function with its parameters bound to actual arguments (i.e., a
closure).  

    PSStencil PSStencilMap(StencilFunctionType stencil,
      PSDomain3D dom, ...)

The `stencil` parameter must be a name of a stencil function. We do
not support specifying functions with function pointers and other
indirect references, and only an actual name with its definition is in
the same compilation unit is accepted so that the
translator can determine the actual stencil function at translation
time. 

The type of the second parameter, `PSDomain3D` is a new data type
to specify a 3-D range, which can be instantiated by the following
intrinsic:

    PSDomain3D PSDomain3DNew(
      size_t x_offset, size_t x_end, 
      size_t y_offset, size_t y_end,
      size_t z_offset, size_t z_end)

Note that similar to grid types, there are 1-D and 2-D
variants of the domain type (i.e., `PSDomain1D` and `PSDomain2D`). The
domain object is used to specify the range of grid points where the stencil function is computed. This parameter can be
used to implement boundary computations that are different from inner
points. 

`PSStencilRun` is used to execute `PSStencil` objects in a
batch manner, as defined as follows: 

    void PSStencilRun(PSStencil s1, PSStencil s2,
      ..., int iter)

It accepts any number of `PSStencil` objects, and executes them in
the given order for `iter` times. However, each stencil function
may be executed in parallel, exhibiting implicit parallelism.

The combination of `PSStencilMap` and `PSStencilRun` forms a
natural unit of code translation. First, it allows for efficient implementation on
multiple-node environments, where we use RPC-based controls, which
forms a global synchronization point between the master node and every
other compute nodes. A single run is translated to an RPC request from
the master node to compute nodes, and the compute nodes execute the stencil
objects in order for the specified times.
Note that each node runs without no RPC-induced global
synchronization. Alternatively, we could design the DSL without run so
that map simply executes a given stencil function; however, a simple
translation would produce a single RPC for each map, which could be a
significant bottleneck when a large number of nodes are used,
resulting in poor performance scalability.

In addition, the combination can be also a unit of code
optimization. Although we have not yet pursued this direction, we plan
to study such optimizations as fusing and reordering of stencil
functions. Explicit grouping of stencil functions would allow for
such aggressive optimizations at translation time. 

Reduction
---------

** THE FOLLOWING IS NOT YET IMPLEMENTED.** It describes the current plan
for data reductions in Physis.	   

Physis provides two methods for reductions. First, grid data can be
reduced to a scalar value with intrinsic `PSReduce`: 

     void PSReduce(T *v, PSGrid g, PSReduceOp op)

Parameter `op` defines a binary associative operation. The predefined 
operations are:

* PS_MAX
* PS_MIN
* PS_SUM
* PS_PROD

The naming and semantics are adopted from the MPI reduction
operations. The result of reduction is stored at the memory location
referenced by the first parameter, `v`, whose type is a pointer to the
element type of the grid parameter.

The same intrinsic also allows for more flexible, in-place data
reduction with a user-defined stencil-like function:

    void PSReduce(T *v, PSReduceOp op, PSReduceKernel func, PSDomain dom, ...)

The first parameter, `v` of type `T *`, is the memory address where
the reduction result is stored. The `PSReduceOp` parameter defines a
reduction operation, which must be one of the predefined ones. The
rest of parameters define mapping of function `func` over the `dom`
domain with optional parameters following the domain parameter. The
function parameter must be a function name unambiguously referencing a
*reduction kernel*, which is a standard C function with the similar
restrictions as the stencil kernel. As the stencil kernel, it is
executed over the given domain with the optional parameters, but,
unlike the stencil kernel, it must return a scalar value of type `T`,
which is then reduced by the given reduction operation `op`.

NOTE: Alternatively, we could design such that the intrinsic directly
returns the result as its return value, and that would look
simpler. We, however, reserve the return value for future extension of
the DSL for error handling.  

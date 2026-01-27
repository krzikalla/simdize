# SIMD Classes - Thinking One Step Further.

**simd_access** is meant as a supplement to `std::experimental::simd` (in short `stdx::simd`).
The aim is to syntactically equalize the loading and saving of simd variables to the loading and saving of
usual variables.
Then loop bodies written for scalar variables can be used without changes for simd variables.

## State of the Art

There are a lot of simd libraries.
They all have in common, that they provide some kind of simd type and overload the usual operators for that type.
Thus, once you have simd variables, you can handle them as usual scalar variables.
However, they also all have in common (as far as I am aware), that obtaining and storing simd variables
is very different from obtaining and storing scalar variables.
In the function
```c++
void times2(int size, double* dest, const double* source)
{
  for (int i = 0; i < size; ++i)
  {
    dest[i] = source[i] * 2;
  }
}
```
you can't just use the line `dest[i] = source[i] * 2;`, if you want to explicitly vectorize this function.
If you want to preserve the look and feel, you need a compiler extension (e.g. `omp simd`).
On the other hand, compiler extensions are often not capable to automatically vectorize complex codes
in the intended way.

## The Core Idea

A typical loop that is suitable for vectorization has three general steps:
```c++
for (int i = 0; i < size; ++i)
{
  // (1) load data associated with i to variables
  // (2) compute results
  // (3) store results to locations associated with i
}
```
In a scalar loop the three steps can and will often be intertwined as in `dest[i] = source[i] * 2;`.
By using a simd data type these three steps become separated, since loading and storing require a more explicit syntax.
Then the second step can already be written in the same way for scalar and vectorized variables with the help of
`stdx::simd`.
C++ achieves this by deducing the matching operators according to the types of the variables created in the first step.

The core idea of the `simd_access` library is now to push the boundary of the type deduction out of the three steps.
Not only the computation, but also load and store operations are deduced.
The deduction is done according to the type of the loop index.
You don't use an integral type as loop index anymore, but a `simd_index`.
Then step (1) and (3) both can deduce the matching operations according to the type of the index.
In addition, it will once again be possible to intertwine all three steps for vectorized code in the way
you are used to.
A simplified version of `times2` (without residual loop handling) then could look as follows:
```c++
void times2(int size, double* dest, const double* source)
{
  using simd_model = stdx::native_simd<double>;
  for (simd_index<simd_model> i = 0; i < size; i += simd_model::size())
  {
    dest[i] = source[i] * 2;
  }
}
```
In this code the expressions `source[i]` and `dest[i]` load and store simd variables according to the
simd model `simd_model`, which is encoded as a template parameter in `simd_index`.
As usual, the multiplication is then also recognized as a simd operation.
Unfortunately, the current C++ language has some shortcomings, so that the above code is not possible yet.
However, the `simd_access` library provides a macro `SIMD_ACCESS` in addition to a `simd_index` type
to enable simd-generic programming.
The library also serves as a basis for discussion for some language extensions, with which the above code ultimately
becomes legal C++.

## The `simd_access` Library

All functions and types of the library reside in the `simd_access` namespace (in short `sa`).
The library contains these general components:
1. The `sa::index`, which represents a simd index to a consecutive sequence of elements. In addition,
`stdx::simd` instantiated with an integral data type can be used as simd index to represents randomly indexed elements.
1. A `SIMD_ACCESS` macro, which can be used to access variables as simd or scalar variables (depending on the index
type passed to it). The variables can also be stored in an array-of-structure (AOS) layout.
1. A `sa::loop` function, which iterates over a given range of indices. Residual iterations are handled properly.
1. An outline for a proposal to enable a global `operator[]` overload.
1. An outline for a proposal to enable a member access overload (`operator.`).


### `sa::index`

The class `sa::index` represents a simd access to a consecutive sequence of elements.
```c++
// SimdModel: Simd type acting as type model.
// IndexType: Type of the index.
template<class SimdModel, class IndexType = size_t>
struct sa::index
{
  // Return SimdModel::size().
  static constexpr size_t size();

  // The index, at which the sequence starts. Thus, the represented sequence is [index_, index_+SimdSize).
  IndexType index_;
};
```
The `SimdModel` template parameter acts as a model, that ultimately determines the vector size of the vectorized
memory accesses.
This parameter replaces the former direct specification of the vector size in order to support platforms on
which the vector size is not a compile-time constant (e.g. ARM SVE).

### The `SIMD_ACCESS` Macro

The `SIMD_ACCESS` macro can be used to rewrite a c++ expression that represents a memory access
into a simd-generic expression.
Let's assume the following typical AOS layout for an array of points:
```c++
struct Point { double x, y };
std::vector<Point> points;
```
The expression `points[i].x` consists of the base array `points`, the index `i` and a member accessor `.x`.
These three components must be passed to the `SIMD_ACCESS` macro separately: `SIMD_ACCESS(points, i, .x)`.
In this way, the memory access becomes simdized, if `i` is `sa:::index` or `stdx:::simd`.
If it is a read access, then `SIMD_ACCESS(points, i, .x)` yields a simd variable containing the values
`points[i].x, points[i+1].x, points[i+2].x ...`.
You can also assign a variable to `SIMD_ACCESS`, in which case the assignment is simdized.
If `i` is an ordinary integral type, then `SIMD_ACCESS` translates to the appropriate scalar access.
```c++
// base: The base array.
// index: The index to the base array. Can be a simd index or an ordinary integral index.
// ...: Possible accessors to data members or elements of a subarray.
#define SIMD_ACCESS(base, index, ...) \
```
`SIMD_ACCESS` can handle a lot of usual memory layouts:
```c++
double direct_array[100]    -> SIMD_ACCESS(direct_array, i)
double sub_array[100][3]    -> SIMD_ACCESS(sub_array, i, [1])       // second element of the sub array
Point pnt_array[100]        -> SIMD_ACCESS(pnt_array, i, .x)        // the x member
Point pnt_sub_array[100][2] -> SIMD_ACCESS(pnt_sub_array, i, [0].x) // the x member of the first element of the sub array
```

#### Shortcomings

`SIMD_ACCESS` expects a contiguous array as `base` with an valid element at index 0.
The type yielded by `SIMD_ACCESS` is not a `stdx::simd`, but provides a type conversion operator and an
assignment operator.
Thus `SIMD_ACCESS` can be used as rvalue as well as lvalue.
In deduced contexts a `stdx::simd` is necessary as rvalue. Use `SIMD_ACCESS_V` in that case.


### The `sa::loop` Function

The `sa::loop` function iterates over a given range of indices and calls a generic functor for each iteration.
The functor is either called with a simd index or for the residual iterations with a scalar index
(by default, see below).
```c++
// Iterates over a consecutive sequence of indices and calls a generic functor for each iteration.
// SimdModel: Simd type acting as type model.
// start Start of the iteration range [start, end).
// end End of the iteration range [start, end).
// fn: Generic functor to be called. Takes one argument, whose type is either `index<SimdModel>`
// or an integral type for the residual iterations.
template<class SimdModel>
void loop(std::integral auto start, std::integral auto end, auto&& fn);
```

In addition to consecutive ranges random index ranges are supported.
```c++
// Iterates over a sequence of random indices and calls a generic functor for each iteration.
// SimdModel: Simd type acting as type model.
// [start, end): Range containing the indices.
// fn: Generic functor to be called. Takes one argument, whose type is either `stdx::simd`
//   or an integral type for the residual iterations.
template<class SimdModel, std::random_access_iterator IteratorType>
void loop(IteratorType start, const IteratorType& end, auto&& fn);
```
Sometimes it is necessary to iterate over a random index range, whereby you also get the linear indices.
In that case you can use `loop_with_linear_index`:
```c++
// Iterates over a sequence of random indices and calls a generic functor for each iteration.
// SimdModel: Simd type acting as type model.
// [start, end): Range containing the indices.
// fn: Generic functor to be called. Takes two arguments. The first is the linear index starting at 0, its
//   type is either `index<SimdModel, size_t>` or `size_t`. The second argument is the indirect index, its type is
//   either `stdx::simd<IntegralType, SimdModel::size()>` or `IntegralType`.
template<class SimdModel, std::random_access_iterator IteratorType>
void loop_with_linear_index(IteratorType start, const IteratorType& end, auto&& fn)
```

Since `SIMD_ACCESS` handles simd indices as well as scalar indices, you can write unified source code for simd and
scalar types.
Thus, now you can write simd and residual iterations in the same way:
```c++
  std::vector<Point> source, result;
  using simd_model = stdx::native_simd<double>;
  sa::loop<simd_model>(0, source.size(), [&](auto i)
    {
      SIMD_ACCESS(result, i, .x) = SIMD_ACCESS(source, i, .x) * 2;
    });
```

Residual iterations occur, if the iteration range is not a multiple of the vector size determined by the simd model type.
`sa::loop` treat indices at the end of the range as residual iterations.
By default, these iterations are called with a scalar index.
However, for integral ranges the `residualLoopPolicy` lets you control this behavior.
If it is set to `VectorResidualLoop`, residual iterations are also called with a simd index.
In that case the simd index might include indices beyond your actual iteration range.
It is up to you how to handle these beyond-the-end indices.
You might have introduced padding, use masking or just know, that there are no residual iterations.
If you use `VectorResidualLoop`, the loop body is not instantiated for scalar indices.
```c++
  double source[64];
  using simd_model = stdx::native_simd<double>;
  // We know, that the iteration range is a multiple of the simd size:
  sa::loop<simd_model>(0, 64, [&](auto i)
    {
      // i is always of type sa::index<simd_model>, so to_simd() is safely available
      auto x = SIMD_ACCESS(source, i).to_simd();
    }, VectorResidualLoop);
```

Note that `sa::loop` can be overloaded for particular simd model types to enable other forms of residual loop handling,
e.g. setting global mask registers (ARM SVE).

### Build Requirements

The lib is header-only. The tests need cmake and a c++20 compliant compiler.
Note, that the tests use class template argument deduction for aggregates, thus for clang a version >= 17 is required.

### Citing

Please consider citing us accordingly in your derived work and publications:
```
% Peer-Reviewed Publication %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Peer reviewed and accepted publication in
%   "2025 IEEE International Conference on High Performance
%    Computing and Communications (HPCC)"
@INPROCEEDINGS{Krzikalla11207437,
  author={Krzikalla, Olaf and Wendler, Johannes and Huismann, Immo},
  booktitle={2025 IEEE International Conference on High Performance Computing and Communications (HPCC)},
  title={Explicit SIMD Vectorization with Modern C++},
  year={2025},
  pages={103-110},
  doi={10.1109/HPCC67675.2025.00033}
}
```

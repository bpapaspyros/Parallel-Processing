// Written by Christian Bienia
// This file aggregates definitions used across all versions of the program

#ifndef __PARTICLES_HPP__
#define __PARTICLES_HPP__ 1

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include <stddef.h>
#if defined(WIN32)
typedef __int64 int64_t;
typedef __int32 int32_t;
typedef __int16 int16_t;
typedef __int8 int8_t;
typedef unsigned __int64 uint64_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int8 uint8_t;
#else
#include <stdint.h>
#endif
#include <math.h>

//Our estimate for a cache line size on this machine
#define CACHELINE_SIZE 128

//Maximum number of particles in a physical cell
#define PARTICLES_PER_CELL 16

static inline int isLittleEndian() {
  union {
    uint16_t word;
    uint8_t byte;
  } endian_test;

  endian_test.word = 0x00FF;
  return (endian_test.byte == 0xFF);
}

//NOTE: Use float variables even for double precision version b/c file format uses float
union __float_and_int {
  uint32_t i;
  float    f;
};

static inline float bswap_float(float x) {
  union __float_and_int __x;

   __x.f = x;
   __x.i = ((__x.i & 0xff000000) >> 24) | ((__x.i & 0x00ff0000) >>  8) |
           ((__x.i & 0x0000ff00) <<  8) | ((__x.i & 0x000000ff) << 24);

  return __x.f;
}

static inline int bswap_int32(int x) {
  return ( (((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |
           (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24) );
}


#if 1
class Vec3
{
public:
  __m128d _xy, _z0;
  double x, y, z;

  Vec3() {}
  Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {
    // loading data on the registers
    _xy = _mm_set_pd( _y, _x);
    _z0 = _mm_set_pd(0.0, _z);
  }

  double  GetLengthSq() const         { 
    // declaring registers for the sse instructions 
    __m128d pres1, pres2;

    // squaring the contents of the vectors
    pres1 = _mm_mul_pd(_xy, _xy);
    pres2 = _mm_mul_pd(_z0, _z0);

    // adding the contents of the two vectors
    pres1 = _mm_add_pd(pres1, pres2);

    // moving y to the top of this register
    pres2 = _mm_set_pd(0.0, pres1[1]);

    // adding the last number
    pres1 = _mm_add_pd(pres1, pres2);

    // adding the two halves of the vector
    return pres1[0];
  }

  double  GetLength() const           { 
    // loading the array to a register
    __m128d res = _mm_set_pd(0.0, GetLengthSq());

    // calculating the square root
    res = _mm_sqrt_pd(res);
  
    return res[0];
  }

  Vec3 &  Normalize()                 { 
    return *this /= GetLength(); 
  }

  bool    operator == (Vec3 const &v) { 
    return (x == v.x) && (y == v.y) && (z += v.z);  
  }

  Vec3 &  operator += (Vec3 const &v) { 
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // loading the s to the registers
    __m128d v1 = _mm_set_pd(v.y, v.x);
    __m128d v2 = _mm_set_pd(  0, v.z);

    // z += v.x
    z = (double)(_mm_add_pd(_z0, v2)[0]);

    // [x y] += [s s]
    v1 = _mm_add_pd(_xy, v1);

    // get the results to the class variables
    x = v1[0]; y = v1[1];

    return *this; 
  }

  Vec3 &  operator -= (Vec3 const &v) { 
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // loading the s to the registers
    __m128d v1 = _mm_set_pd(v.y, v.x);
    __m128d v2 = _mm_set_pd(  0, v.z);

    // z -= v.x
    z = (double)(_mm_sub_pd(_z0, v2)[0]);

    // [x y] -= [s s]
    v1 = _mm_sub_pd(_xy, v1);

    // get the results to the class variables
    x = v1[0]; y = v1[1];

    return *this; 
  }

  Vec3 &  operator *= (double s)      { 
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // loading the s to the registers
    __m128d m = _mm_set_pd(s, s);

    // z *= s
    z = (double)(_mm_mul_pd(_z0, m)[0]);

    // [x y] * [s s]
    m = _mm_mul_pd(_xy, m);

    // get the results to the class variables
    x = m[0]; y = m[1];

    return *this; 
  }

  Vec3 &  operator /= (double s)      {
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // load 2 of the registers with data
    __m128d _s = _mm_set_pd(s, s);
    __m128d _ones = _mm_set_pd(1.f, 1.f);

    /* we could avoid contsructing this *
     * fraction but it is kept for      *
     * compatibility reasons, regarding *
     * previous implementations         */
      // divide them so that we create
      // the 1/s fraction
    __m128d pres1 = _mm_div_pd(_ones, _s);
    __m128d pres2 = pres1;
    
    // complete the multiplication
    pres1 = _mm_mul_pd(_xy, pres1);
    pres2 = _mm_mul_pd(_z0, pres2);

    // get the results to the class variables
    x = pres1[0]; y = pres1[1]; z = pres2[0];

    return *this; 
  }

  Vec3    operator + (Vec3 const &v) const    { 
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // load the arrays to the registers
    __m128d res1 = _mm_set_pd(v.y, v.x);
    __m128d res2 = _mm_set_pd(  0, v.z);

    // make the additions
    res1 = _mm_add_pd(_xy, res1);
    res2 = _mm_add_pd(_z0, res2);

    // return a new object
    return Vec3(res1[0], res1[1], res2[0]); 
  }
  
  Vec3    operator + (double const &f) const  { 
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // load the arrays to the registers
    __m128d res1 = _mm_set_pd(f, f);
    __m128d res2 = res1;

    // make the additions
    res1 = _mm_add_pd(_xy, res1);
    res2 = _mm_add_pd(_z0, res2);

    // return a new object
    return Vec3(res1[0], res1[1], res2[0]); 
  }

  Vec3    operator - () const                 { 
    return Vec3(-x, -y, -z); 
  }

  Vec3    operator - (Vec3 const &v) const    { 
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // load the arrays to the registers
    __m128d res1 = _mm_set_pd(v.y, v.x);
    __m128d res2 = _mm_set_pd(  0, v.z);

    // make the subtractions
    res1 = _mm_sub_pd(_xy, res1);
    res2 = _mm_sub_pd(_z0, res2);

    // return a new object
    return Vec3(res1[0], res1[1], res2[0]); 
  }

  Vec3    operator * (double s) const         { 
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // load the arrays to the registers
    __m128d res1 = _mm_set_pd(s, s);
    __m128d res2 = _mm_set_pd(0, s);

    // make the multiplication
    res1 = _mm_mul_pd(_xy, res1);
    res2 = _mm_mul_pd(_z0, res2);

    // return a new object
    return Vec3(res1[0], res1[1], res2[0]);   
  }

  Vec3    operator / (double s) const         { 
    // loading our variables to the registers
    __m128d _xy = _mm_set_pd(y, x);
    __m128d _z0 = _mm_set_pd(0, z);

    // load 2 of the registers with data
    __m128d _s = _mm_set_pd(s, s);
    __m128d _ones = _mm_set_pd(1.f, 1.f);

    /* we could avoid contsructing this *
     * fraction but it is kept for      *
     * compatibility reasons, regarding *
     * previous implementations         */
      // divide them so that we create
      // the 1/s fraction
    __m128d pres1 = _mm_div_pd(_ones, _s);
    __m128d pres2 = pres1;
    
    // complete the multiplication
    pres1 = _mm_mul_pd(_xy, pres1);
    pres2 = _mm_mul_pd(_z0, pres2);

    // returning the object
    return Vec3(pres1[0], pres1[1], pres2[0]); 
  }

  double  operator * (Vec3 const &v) const    { 
    // load the arrays to the registers
    __m128d res1 = _mm_set_pd(v.y, v.x);
    __m128d res2 = _mm_set_pd(0.0, v.z);

    // make the multiplications
    res1 = _mm_mul_pd(_xy, res1);
    res2 = _mm_mul_pd(_z0, res2);

    // add the two vectors (scalar)
    res1 = _mm_add_pd(res1, res2);

    // move y*v.y to the other register
    std::swap(res2[0], res1[1]);

    // return the sum
    return (_mm_add_pd(res1, res2)[0]); 
  }
};

#else 
class Vec3
{
public:
  double x, y, z;

  Vec3() {}
  Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

  double  GetLengthSq() const         { return x*x + y*y + z*z; }
  double  GetLength() const           { return sqrtf(GetLengthSq()); }
  Vec3 &  Normalize()                 { return *this /= GetLength(); }

  bool    operator == (Vec3 const &v) { return (x == v.x) && (y == v.y) && (z += v.z); }
  Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
  Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
  Vec3 &  operator *= (double s)      { x *= s;  y *= s; z *= s; return *this; }
  Vec3 &  operator /= (double s)      { double tmp = 1.f/s; x *= tmp;  y *= tmp; z *= tmp; return *this; }

  Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
  Vec3    operator + (double const &f) const  { return Vec3(x+f, y+f, z+f); }
  Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
  Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
  Vec3    operator * (double s) const         { return Vec3(x*s, y*s, z*s); }
  Vec3    operator / (double s) const         { double tmp = 1.f/s; return Vec3(x*tmp, y*tmp, z*tmp); }

  double  operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
};
#endif

////////////////////////////////////////////////////////////////////////////////

// We define two Cell structures - one helper structure without padding and one
// "real" structure with padding to be used by the program. The helper structure
// is needed because compilers can insert different amounts of auto-generated
// padding and we need to know the exact amount to calculate the cache line
// padding accurately. By having two structures we can reference that amount
// for the padding calculations. Both structures must have the same amount
// of payload data, which we check with an assert in the program. Make
// sure to keep both structures in sync.

// NOTE: Please note the difference between a logical cell and a physical
// cell. A logical cell corresponds to a 3D region in space and contains all
// the particles in that region. A physical cell is the Cell structure defined
// below. Each logical cell is implemented of a linked list of physical cells.

//Actual particle data stored in the cells
#define CELL_CONTENTS \
  Vec3 p[PARTICLES_PER_CELL]; \
  Vec3 hv[PARTICLES_PER_CELL]; \
  Vec3 v[PARTICLES_PER_CELL]; \
  Vec3 a[PARTICLES_PER_CELL]; \
  double density[PARTICLES_PER_CELL];

//Helper structure for padding calculation, not used directly by the program
struct Cell_aux {
  CELL_CONTENTS
  Cell_aux *next;
  //dummy variable so we can reference the end of the payload data
  char padding;
};

//Real Cell structure
struct Cell {
  CELL_CONTENTS
  Cell *next;
  //padding to force cell size to a multiple of estimated cache line size
  char padding[CACHELINE_SIZE - (offsetof(struct Cell_aux, padding) % CACHELINE_SIZE)];
  Cell() { next = NULL; }
};

////////////////////////////////////////////////////////////////////////////////

static const float pi = 3.14159265358979;

static const float parSize = 0.0002;
static const float epsilon = 1e-10;
static const float stiffnessPressure = 3.0;
static const float stiffnessCollisions = 30000.0;
static const float damping = 128.0;
static const float viscosity = 0.4;

static const float doubleRestDensity = 2000.0;
static const float kernelRadiusMultiplier = 1.695;
static const Vec3 externalAcceleration(0.0, -9.8, 0.0);
static const Vec3 domainMin(-0.0650000000, -0.080000000, -0.0650000000);
static const Vec3 domainMax(0.0650000000, 0.10000000, 0.0650000000);
static const float Zero = 0.0;
//Constants for file I/O
#define FILE_SIZE_INT 4
#define FILE_SIZE_FLOAT 4

#endif //__PARTICLES_HPP__

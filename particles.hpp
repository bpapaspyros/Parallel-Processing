// Written by Christian Bienia
// This file aggregates definitions used across all versions of the program

#ifndef __PARTICLES_HPP__
#define __PARTICLES_HPP__ 1

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


class Vec3
{
public:
  float x, y, z;

  Vec3() {}
  Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

  float  GetLengthSq() const         { return x*x + y*y + z*z; }
  float  GetLength() const           { return sqrtf(GetLengthSq()); }
  Vec3 &  Normalize()                 { return *this /= GetLength(); }

  bool    operator == (Vec3 const &v) { return (x == v.x) && (y == v.y) && (z += v.z); }
  Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
  Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
  Vec3 &  operator *= (float s)      { x *= s;  y *= s; z *= s; return *this; }
  Vec3 &  operator /= (float s)      { float tmp = 1.f/s; x *= tmp;  y *= tmp; z *= tmp; return *this; }

  Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
  Vec3    operator + (float const &f) const  { return Vec3(x+f, y+f, z+f); }
  Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
  Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
  Vec3    operator * (float s) const         { return Vec3(x*s, y*s, z*s); }
  Vec3    operator / (float s) const         { float tmp = 1.f/s; return Vec3(x*tmp, y*tmp, z*tmp); }

  float  operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
};



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

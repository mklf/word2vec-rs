//
//

#ifndef FMAT_VEC_ARITH_H
#define FMAT_VEC_ARITH_H
#define data_t float
#define VBYTES  32
#define  VSIZE  VBYTES/ sizeof(data_t)

#include <stdio.h>
typedef data_t vec_t __attribute__ ((vector_size(VBYTES)));
data_t dot_product(const data_t* __restrict__ a,const data_t* __restrict__ b,size_t length);
data_t simd_dot_product(const data_t* __restrict__ a,const data_t* __restrict__ b,size_t size);
data_t simd_dot_product_x4(const data_t* __restrict__ a,const data_t* __restrict__ b,size_t size);

void  saxpy(data_t*  dst, const data_t* source, const data_t scale,size_t size);

void  saxpy_x4(data_t* __restrict__ dst, const data_t* __restrict__ source, const data_t scale,size_t size);

void  simd_saxpy(data_t* __restrict__ dst, const data_t* __restrict__ source, data_t scale, size_t  size);
#endif //FMAT_VEC_ARITH_H

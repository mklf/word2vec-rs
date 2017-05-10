#include "vec_arith.h"
#include <stdio.h>

// data not owned by this vector
data_t dot_product(const data_t* __restrict__ a,const data_t* __restrict__ b,size_t length){
    data_t res0 = 0;
    data_t res1 = 0;
    size_t i=0;
    for(i=0;i<length-2;i+=2){
        res0 = res0+ a[i] * b[i];
        res1 = res1+ a[i+1] * b[i+1];
    }
    res0+=res1;
    for (;i<length;++i) {
        res0 +=a[i] * b[i];
    }
    return res0;
}
data_t simd_dot_product(const data_t* __restrict__ a,const data_t* __restrict__ b,size_t size){
    vec_t accum;
    data_t result = 0;
    for (int i = 0; i < VSIZE; ++i) {
        accum[i] = 0;
    }
    while(((long)a)% VBYTES &&size){
        result +=*a++ * *b++;
        size--;
    }
    while(size>=VSIZE){
        vec_t a_chunk = *((vec_t*)a);
        vec_t b_chunk = *((vec_t*)b);
        accum = accum + (a_chunk * b_chunk);
        a += VSIZE;
        b += VSIZE;
        size -=VSIZE;
    }
    while(size){
        result += *a++ * *b++;
        size--;
    }
    for (int j = 0; j < VSIZE; ++j) {
        result +=accum[j];
    }
    return result;
}
data_t simd_dot_product_x4(const data_t* __restrict__ a,const data_t* __restrict__ b,size_t size){
    vec_t accum0,accum1,accum2,accum3;
    size_t cnt = size;
    data_t result = 0;
    for (int i = 0; i < VSIZE; ++i) {
        accum0[i] = 0;accum1[i] = 0;
        accum2[i] = 0;accum3[i] = 0;
    }
    while(((long)a)% VBYTES &&cnt){
        result +=*a++ * *b++;
        cnt--;
    }
    while(cnt>=4*VSIZE){
        vec_t a_chunk = *((vec_t*)a);
        vec_t b_chunk = *((vec_t*)b);
        accum0 = accum0 + (a_chunk * b_chunk);
        a += VSIZE;b += VSIZE;

        a_chunk = *((vec_t*)a);
        b_chunk = *((vec_t*)b);
        accum1 = accum1 +(a_chunk* b_chunk);
        a += VSIZE;b += VSIZE;

        a_chunk = *((vec_t*)a);
        b_chunk = *((vec_t*)b);
        accum2 = accum2 +(a_chunk* b_chunk);
        a += VSIZE;b += VSIZE;

        a_chunk = *((vec_t*)a);
        b_chunk = *((vec_t*)b);
        accum3 = accum3 +(a_chunk* b_chunk);
        a += VSIZE;b += VSIZE;

        cnt -=4*VSIZE;
    }
    while(cnt){
        result += *a++ * *b++;
        cnt--;
    }
    accum0 +=accum1;
    accum2 +=accum3;
    accum0 +=accum2;

    for (int j = 0; j < VSIZE; ++j) {
        result +=accum0[j];
    }
    return result;
}

//
void  saxpy(data_t*  dst, const data_t* source, const data_t scale,size_t size){
    for (size_t i = 0; i < size; i++) {
        dst[i] = dst[i] + scale * source[i];
    }
}

void  saxpy_x4(data_t* __restrict__ dst, const data_t* __restrict__ source, const data_t scale,size_t size){
    size_t i=0;
    for (i = 0; i < size-4; i+=4) {
        dst[i] += scale * source[i];
        dst[i+1] += scale* source[i+1];
        dst[i+2] += scale* source[i+2];
        dst[i+3] += scale* source[i+3];
    }
    for (; i < size; ++i) {
        dst[i] += scale * source[i];
    }

}

void  simd_saxpy(data_t* __restrict__ dst, const data_t* __restrict__ source, data_t scale, size_t  size){
    while(((long)dst)% VBYTES &&size) {
        *dst = *dst + scale * *source;
        dst++;source++;
        size--;
    }
    vec_t s,t;

    for (int i = 0; i < VSIZE; ++i) {
        s[i] = scale;
    }
    while(size>=VSIZE){
        vec_t a_chunk = *((vec_t*)source);
        t = s*a_chunk;
        for (int i = 0; i < VSIZE; ++i) {
            *dst++ = t[i];
        }
        size -=VSIZE;
        source+=VSIZE;
    }
    for (; size ; --size) {
        *dst++ = scale * *source++;
    }

}
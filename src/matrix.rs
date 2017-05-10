use rand;
#[cfg(feature="blas")]
use blas_sys::c;
use rand::distributions::{IndependentSample, Range};
use std::cell::UnsafeCell;

use simd_dot_product;
use saxpy;

#[derive(Debug)]
pub struct MatrixWrapper {
    pub inner: UnsafeCell<Matrix>,
}
unsafe impl Sync for MatrixWrapper {}

#[derive(Debug)]
pub struct Matrix {
    row_size: usize,
    mat: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, row_size: usize) -> Matrix {
        Matrix {
            mat: vec![0f32;row_size * rows],
            row_size: row_size,
        }
    }


    #[allow(unused_mut)]
    pub fn norm_self(&mut self) {
        let mut ptr = self.mat.as_mut_ptr();
        for i in 0..self.mat.len() / self.row_size {
            let basei = self.row_size as isize * i as isize;
            let n = self.norm(i);
            for j in 0..self.row_size {
                unsafe { (*ptr.offset(basei + j as isize)) /= n };
            }
        }
    }
    pub fn make_send(self) -> MatrixWrapper {
        MatrixWrapper { inner: UnsafeCell::new(self) }
    }
    pub fn unifrom(&mut self, bound: f32) {
        let between = Range::new(-bound, bound);
        let mut rng = rand::thread_rng();
        for v in &mut self.mat {
            *v = between.ind_sample(&mut rng);
        }

    }
    #[inline(always)]
    pub fn zero(&mut self) {
        for v in &mut self.mat.iter_mut() {
            *v = 0f32;
        }
    }

    #[inline(always)]
    pub fn add_row(&mut self, vec: *mut f32, i: usize, mul: f32) {

        unsafe {
            //simd_saxpy(
            saxpy(
                self.mat.get_unchecked_mut(i*self.row_size),
                vec,
                mul,
                self.row_size
            );
        }
    }


    #[inline(always)]
    pub fn dot_row(&mut self, vec: *const f32, i: usize) -> f32 {
        unsafe {
            simd_dot_product(
            //dot_product(
                self.mat.get_unchecked(i* self.row_size),
                vec,
                self.row_size
            )
        }

    }
    #[inline(always)]
    pub fn dot_two_row(&mut self, i: usize, j: usize) -> f32 {
            unsafe {
                simd_dot_product(
                //dot_product(
                    self.mat.get_unchecked(i* self.row_size),
                    self.mat.get_unchecked(j* self.row_size),
                    self.row_size
                )

            }
    }


    #[allow(unused_mut)]
    pub fn norm(&mut self, i: usize) -> f32 {
        let basei = self.row_size as isize * i as isize;
        let mut n = 0.;
        let mut ptr = self.mat.as_mut_ptr();
        for t in 0..self.row_size as isize {
            n += unsafe { (*ptr.offset(basei + t)).powf(2.0) };
        }
        n.sqrt()
    }


    #[inline(always)]
    pub fn get_row(&mut self, i: usize) -> *mut f32 {
        unsafe { self.mat.get_unchecked_mut(i * self.row_size) }
    }
    #[inline(always)]
    pub fn get_row_unmod(&self, i: usize) -> *const f32 {
        unsafe { self.mat.get_unchecked(i * self.row_size) }
    }

}

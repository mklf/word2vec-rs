use rand;
#[cfg(feature="blas")]
use blas_sys::c;
use rand::distributions::{IndependentSample, Range};
use std::cell::UnsafeCell;

#[derive(Debug)]
pub struct MatrixWrapper {
    pub inner: UnsafeCell<Matrix>,
}
unsafe impl Sync for MatrixWrapper {}

#[derive(RustcEncodable,RustcDecodable,PartialEq,Debug)]
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
    #[cfg(feature="blas")]
    #[inline(always)]
    pub fn sgemv(&self, vec: *const f32, out: *mut f32) {
        unsafe {
            c::cblas_sgemv(c::CblasRowMajor,
                           c::CblasNoTrans,
                           (self.mat.len() / self.row_size) as i32,
                           self.row_size as i32,
                           1.,
                           self.mat.as_ptr(),
                           self.row_size as i32,
                           vec,
                           1,
                           1.,
                           out,
                           1);
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

    #[cfg(feature="blas")]
    #[inline(always)]
    pub fn add_row(&mut self, vec: *mut f32, i: usize, mul: f32) {
        unsafe {
            c::cblas_saxpy(self.row_size as i32,
                           mul,
                           vec,
                           1,
                           self.mat.get_unchecked_mut(i * self.row_size),
                           1);
        }
    }
    #[cfg(not(feature="blas"))]
    #[inline(always)]
    #[allow(unused_mut)]
    pub fn add_row(&mut self, vec: *const f32, i: usize, mul: f32) {
        let base = i as isize * self.row_size as isize;
        let mut ptr = self.mat.as_mut_ptr();
        for t in 0..self.row_size as isize {
            unsafe {
                *ptr.offset(base + t) += mul * (*vec.offset(t));
            }
        }
    }

    #[cfg(feature="blas")]
    #[inline(always)]
    pub fn dot_row(&mut self, vec: *const f32, i: usize) -> f32 {
        unsafe {
            c::cblas_sdot(self.row_size as i32,
                          self.mat.get_unchecked(i * self.row_size),
                          1,
                          vec,
                          1)
        }
    }
    #[cfg(feature="blas")]
    #[inline(always)]
    pub fn dot_two_row(&mut self, i: usize, j: usize) -> f32 {
        unsafe {
            c::cblas_sdot(self.row_size as i32,
                          self.mat.get_unchecked(i * self.row_size),
                          1,
                          self.mat.get_unchecked(j * self.row_size),
                          1)
        }
    }
    #[cfg(not(feature="blas"))]
    #[inline(always)]
    pub fn dot_two_row(&mut self, i: usize, j: usize) -> f32 {
        let mut sum = 0f32;
        let basei = self.row_size as isize * i as isize;
        let basej = self.row_size as isize * j as isize;
        let ptr = self.mat.as_ptr();
        for t in 0..self.row_size as isize {
            unsafe {
                sum += *ptr.offset(basei + t) * *ptr.offset(basej + t);
            };
        }
        sum
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

    #[cfg(not(feature="blas"))]
    #[inline(always)]
    pub fn dot_row(&mut self, vec: *const f32, i: usize) -> f32 {
        let mut sum = 0f32;
        let base = self.row_size as isize * i as isize;
        let ptr = self.mat.as_ptr();
        for t in 0..self.row_size as isize {
            unsafe {
                sum += *ptr.offset(base + t) * (*vec.offset(t as isize));
            };
        }
        sum

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

use rand;
use rand::distributions::{IndependentSample, Range};
use std::cell::UnsafeCell;

pub struct MatrixWrapper {
    pub inner: UnsafeCell<Matrix>,
}
unsafe impl Sync for MatrixWrapper {}

pub struct Matrix {
    nrows: usize,
    ncols: usize,
    mat: Vec<f32>,
}
impl Matrix {
    pub fn new(ncols: usize, nrows: usize) -> Matrix {
        Matrix {
            mat: vec![0f32;ncols * nrows],
            ncols: ncols,
            nrows: nrows,
        }
    }

    pub fn clone(self) -> MatrixWrapper {
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
        for t in 0..self.nrows {
            unsafe {
                *self.mat.get_unchecked_mut(i * self.nrows + t) += mul * (*vec.offset(t as isize));
            }
        }
    }
    #[inline(always)]
    pub fn dot_row(&mut self, vec: *mut f32, i: usize) -> f32 {
        let mut sum = 0f32;
        for t in 0..self.nrows {
            unsafe {
                sum += *self.mat.get_unchecked(i * self.nrows + t) * (*vec.offset(t as isize));
            };
        }
        sum
    }
    #[inline(always)]
    pub fn get_row(&mut self, i: usize) -> *mut f32 {
        unsafe { self.mat.get_unchecked_mut(i * self.nrows) }
    }
}

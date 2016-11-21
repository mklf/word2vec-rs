use {Matrix, Dict, MatrixWrapper};
use std::io::{BufWriter, stdout, stderr};
use std::io::prelude::*;
use std::fs::File;
use std::mem;
#[cfg(feature="blas")]
use blas_sys::c;
use std::cmp::Ordering;
use std::sync::Arc;
pub struct Word2vec {
    syn0: Arc<MatrixWrapper>,
    syn1neg: Arc<MatrixWrapper>,
    dim: usize,
    dict: Arc<Dict>,
}

impl Word2vec {
    pub fn new(syn0: Arc<MatrixWrapper>,
               syn1neg: Arc<MatrixWrapper>,
               dim: usize,
               dict: Arc<Dict>)
               -> Word2vec {
        Word2vec {
            syn0: syn0,
            syn1neg: syn1neg,
            dim: dim,
            dict: dict,
        }
    }
    pub fn norm_self(&mut self) {
        unsafe { (*self.syn0.as_ref().inner.get()).norm_self() };
    }
    #[cfg(feature="blas")]
    #[inline(always)]
    pub fn most_similar(&self, word: &str, topn: Option<usize>) -> Vec<(f32, String)> {
        let mut vec = vec![0.;self.dict.nsize()];
        let c = self.dict.get_idx(word);
        unsafe {
            let mut syn0 = self.syn0.inner.get();
            let row = (*syn0).get_row(c);
            (*syn0).sgemv(row, vec.as_mut_ptr());
        }
        let mut sorted = Vec::new();
        for i in 0..vec.len() {
            sorted.push((vec[i], self.dict.get_word(i)));
        }
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        sorted
    }
    pub fn save(&self, filename: &str) {
        use std::fmt;
        let size = self.dict.nsize();
        let mut file = File::create(filename).unwrap();
        let mut meta = Vec::new();

        write!(&mut meta, "{} {}\n", size, self.dim);
        file.write_all(&meta);
        let syn0 = self.syn0.inner.get();
        let start = unsafe { (*syn0).get_row(0) };
        for i in 0..size {
            file.write_all(&self.dict.get_word(i).into_bytes()[..]);
            for j in 0..self.dim {
                unsafe {
                    let s = format!(" {}", *start.offset((i * self.dim + j) as isize));
                    file.write(&s.into_bytes()[..]);
                }
            }
            file.write(b"\n");
        }
    }
}
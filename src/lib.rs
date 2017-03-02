extern crate rand;
#[cfg(feature="blas")]
extern crate blas_sys;
extern crate time;
extern crate libc;
extern crate rustc_serialize;
extern crate bincode;
mod model;
pub use model::Model;

pub mod dictionary;
pub use dictionary::Dict;
pub mod matrix;
pub use matrix::Matrix;
mod utils;
pub use utils::{Argument, parse_arguments, Command};

pub mod train;
pub use train::train;
const SIGMOID_TABLE_SIZE: usize = 512;
const MAX_SIGMOID: f32 = 8f32;
const NEGATIVE_TABLE_SIZE: usize = 10000000;
const LOG_TABLE_SIZE: usize = 512;

mod w2v;
pub use w2v::Word2vec;
pub use utils::W2vError;
#[macro_use]
extern crate clap;

mod ffi;
pub use ffi::*;
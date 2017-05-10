extern crate rand;
extern crate time;
extern crate libc;
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

use libc::size_t;

#[link(name = "vec_arith")]
extern {
    pub fn simd_dot_product_x4(a: *const f32,b:*const f32,
                            length: size_t )->f32;
    pub fn simd_saxpy(dst:* mut f32, source:*const f32,scale:f32,size:size_t);

    pub fn saxpy_x4(dst:* mut f32, source:*const f32,scale:f32,size:size_t);

    pub fn simd_dot_product(a: *const f32,b:*const f32,
                            length: size_t )->f32;

    pub fn dot_product(a: *const f32,b:*const f32,
                            length: size_t )->f32;


    pub fn saxpy(dst:* mut f32, source:*const f32,scale:f32,size:size_t);
    /*
    fn snappy_compress(input: *const u8,
                       input_length: size_t,
                       compressed: *mut u8,
                       compressed_length: *mut size_t) -> c_int;
    fn snappy_uncompress(compressed: *const u8,
                         compressed_length: size_t,
                         uncompressed: *mut u8,
                         uncompressed_length: *mut size_t) -> c_int;
    fn snappy_max_compressed_length(source_length: size_t) -> size_t;
    fn snappy_uncompressed_length(compressed: *const u8,
                                  compressed_length: size_t,
                                  result: *mut size_t) -> c_int;
    fn snappy_validate_compressed_buffer(compressed: *const u8,
                                         compressed_length: size_t) -> c_int;
    */
}
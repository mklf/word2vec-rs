extern crate rand;
#[cfg(feature="blas")]
extern crate blas_sys;
extern crate time;
extern crate libc;
mod model;
pub use model::Model;

pub mod dictionary;
pub use dictionary::Dict;
pub mod matrix;
pub use matrix::Matrix;
pub mod utils;
pub use utils::parse_arguments;
pub use utils::Argument;

pub mod train;
pub use train::train;
const SIGMOID_TABLE_SIZE: usize = 512;
const MAX_SIGMOID: f32 = 8f32;
const NEGATIVE_TABLE_SIZE: usize = 10000000;
const LOG_TABLE_SIZE: usize = 512;



#[macro_use]
extern crate clap;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
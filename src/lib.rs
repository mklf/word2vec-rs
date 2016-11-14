#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}


mod model;
pub use model::Model;
pub use model::ModelBuilder;

pub mod dictionary;
pub use dictionary::Dict;
pub mod matrix;
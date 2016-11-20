extern crate word2vec;
use word2vec::{train, parse_arguments};
use std::env::args;

fn main() {
    let args_str = args().collect::<Vec<String>>();
    let arguments = parse_arguments(&args_str);
    println!("{:?}", arguments);
    train(&arguments);

    // more program logic goes here...
}
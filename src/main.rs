extern crate word2vec;
use word2vec::dictionary::*;
fn main() {
    println!("{:?}", "hello world");
    let mut d = Dict::new();
    
    d.read_from_file("../wiki_seg_5e4.txt");
}
extern crate word2vec;
use word2vec::dictionary::*;
use std::io;
fn main() {
    println!("{:?}", "hello world");
    let mut d = Dict::new();

    d.read_from_file("../wiki_seg_5e4.txt");
    let mut s = String::new();
    std::io::stdin().read_line(&mut s);
}
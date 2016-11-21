extern crate word2vec;
use word2vec::{train, parse_arguments};
use std::env::args;
use std::io;
fn main() {
    let args_str = args().collect::<Vec<String>>();
    let arguments = parse_arguments(&args_str);
    println!("{:?}", arguments);
    let w2v = train(&arguments);
     #[cfg(feature="blas")]
    loop {
        print!("input: query:");
        let mut s = String::new();
        io::stdin().read_line(&mut s).unwrap();
        println!("before similar");
        let similar = w2v.most_similar(s.trim(), Some(10));
        println!("before similar");
        for ref k in similar[..10].iter() {
            println!("{},{}", k.0, k.1);
        }
        s.clear();
    }
}
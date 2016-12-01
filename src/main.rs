extern crate word2vec;
use word2vec::{train, parse_arguments, Command};
use std::env::args;
use std::io;
fn main() {

    let args_str = args().collect::<Vec<String>>();
    let arguments = parse_arguments(&args_str);
    if arguments.is_err() {
        println!("argument error use --help for help");
        return;
    }
    let arguments = arguments.unwrap();
    if arguments.command == Command::Train {
        let w2v = train(&arguments).unwrap();
        w2v.save_vectors(&arguments.output);

    }
    if arguments.command == Command::Test {
        let w2v = word2vec::Word2vec::load_from(&arguments.input);
        if let Err(e) = w2v {
            println!("{}", e);
            return;
        }
        let mut w2v = w2v.unwrap();
        w2v.norm_self();
     #[cfg(feature="blas")]
        loop {
            println!("input query:");
            let mut s = String::new();
            io::stdin().read_line(&mut s).unwrap();
            let similar = w2v.most_similar(s.trim(), Some(10));
            println!("before similar");
            for ref k in similar[..10].iter() {
                println!("{},{}", k.0, k.1);
            }
            s.clear();
        }
    }
}
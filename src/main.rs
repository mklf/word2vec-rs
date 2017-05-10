extern crate word2vec;
use word2vec::{train, parse_arguments, Command};
use std::env::args;
fn main() {

    let args_str = args().collect::<Vec<String>>();
    let arguments = parse_arguments(&args_str);
    if arguments.is_err() {
        println!("argument error --help for help");
        return;
    }
    let arguments = arguments.unwrap();

    if arguments.command == Command::Train {
        let w2v = train(&arguments).expect("error enconter when training");
        w2v.save_vectors(&arguments.output).expect("error save vectors");

    }

}
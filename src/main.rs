extern crate word2vec;
use word2vec::ModelBuilder;
fn main() {
    println!("{:?}", "hello world");
    let mut model = ModelBuilder::new().finallize();
    model.train("../wiki_seg_5e4.txt", 5);
    let mut s = String::new();
    println!("type to exit");
    std::io::stdin().read_line(&mut s).unwrap();

}
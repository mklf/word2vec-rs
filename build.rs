extern crate gcc;
fn main(){
    gcc::Config::new().flag("-O3").flag("-mavx").flag("-std=c99").
        file("vec_arith.c").
        compile("libvec_arith.a");
}
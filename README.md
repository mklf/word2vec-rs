# Word2Vec-rs

Word2Vec-rs is a fast implemention of word2vec's skip-gram algorithm.

A simple benchmark on a 200M english corpus:
4 thread:
|tool | words per sec| memory  |
|---|---|---|
|word2vec| 332k  | 616M|
|gensim  |311k   | 249M|
|word2vec-rs|641K|197M |

8 thread:
|tool | words per sec| memory  |
|---|---|---|
|word2vec| 611k  | 616M|
|gensim  |539k   | 266M|
|word2vec-rs|995k|203M |

# Building
word2vec-rs is written in Rust, so you need a [Rust installation](https://www.rust-lang.org/) in order to compile it( It's super easy).
Rust version 1.14 or newer is tested.Building is easy:
```
git clone https://github.com/mklf/word2vec-rs
cd word2vec-rs
cargo build --release --features=blas
./target/release/word2vec train --help
```
# Running
```
./target/release/word2vec train input_file_path output_path
```
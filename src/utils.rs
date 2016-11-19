#[derive(Debug,Clone)]
pub struct Argument {
    pub input: String,
    pub output: String,
    pub lr: f32,
    pub dim: usize,
    pub win: usize,
    pub epoch: u32,
    pub neg: usize,
    pub nthreads: u32,
    pub min_count: usize,
    pub threshold: f32,
    pub lr_update: u32,
    pub command: String,
}

pub fn parse_arguments<'a>(args: &'a Vec<String>) -> Argument {
    let matches = clap_app!(word2vec =>
        (version: "1.0")
        (author: "Frank Lee <golifang1234@gmail.com>")
        (about: "word2vec implemention for rust")

       (@subcommand train =>
            (about: "train model")
            (version: "0.1")
         //argument
        (@arg input: +required "input corpus file path")
        (@arg output: +required "file name to save params")
        //options
        (@arg win: --win +takes_value "window size(5)")
        (@arg neg: --neg +takes_value "negative sampling size(5)")
        (@arg lr: --lr +takes_value "learning rate(0.05)")
        (@arg lr_update: --lr_update +takes_value "learning rate update rate(100)")
        (@arg dim: --dim +takes_value "size of word vectors(100)")
        (@arg epoch: --epoch +takes_value "number of epochs(5)")
        (@arg min_count: --min_count +takes_value "number of word occurences(5)")
        (@arg nthreads: --thread +takes_value "number of threads(12)")
        (@arg threshold: --threshold +takes_value "sampling threshold(1e-4)")
        )
    )
        .get_matches_from(args);

    let train_info = matches.subcommand_matches("train").unwrap();
    let input = train_info.value_of("input").unwrap();
    let output = train_info.value_of("output").unwrap();
    let win = train_info.value_of("win").unwrap_or("5").parse::<usize>().unwrap();
    let neg = train_info.value_of("neg").unwrap_or("5").parse::<usize>().unwrap();
    let lr = train_info.value_of("lr").unwrap_or("0.05").parse::<f32>().unwrap();
    let lr_update = train_info.value_of("lr_update").unwrap_or("100").parse::<u32>().unwrap();
    let vector_size = train_info.value_of("dim").unwrap_or("100").parse::<usize>().unwrap();
    let epoch = train_info.value_of("epoch").unwrap_or("5").parse::<u32>().unwrap();
    let min_count = train_info.value_of("min_count").unwrap_or("5").parse::<usize>().unwrap();
    let nthreads = train_info.value_of("nthreads").unwrap_or("12").parse::<u32>().unwrap();
    let threshold = train_info.value_of("threshold").unwrap_or("1e-4").parse::<f32>().unwrap();
    Argument {
        input: input.to_string(),
        output: output.to_string(),
        lr: lr,
        dim: vector_size,
        win: win,
        epoch: epoch,
        neg: neg,
        nthreads: nthreads,
        min_count: min_count,
        threshold: threshold,
        lr_update: lr_update,
        command: "train".to_string(),
    }
}
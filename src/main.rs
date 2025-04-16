mod codegen;

use codegen::CodeGen;
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use std::error::Error;
use std::{env, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let filename = env::args().nth(1).ok_or("file not provided")?;
    let bf_code = fs::read_to_string(filename)?;

    let context = Context::create();
    let codegen = CodeGen::new(&context, &bf_code, None, OptimizationLevel::Default)?;

    let main = codegen.get_jit_fn()?;

    unsafe {
        main.call();
    }

    Ok(())
}

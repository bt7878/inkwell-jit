use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::types::{ArrayType, IntType};
use inkwell::values::{FunctionValue, GlobalValue, PointerValue};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

use std::error::Error;

type BfFunc = unsafe extern "C" fn() -> i32;

const MAIN: &str = "main";

enum OpType {
    Inc,
    Dec,
}

struct Types<'ctx> {
    i8_type: IntType<'ctx>,
    i32_type: IntType<'ctx>,
    mem_type: ArrayType<'ctx>,
}

struct IO<'ctx> {
    getchar: FunctionValue<'ctx>,
    putchar: FunctionValue<'ctx>,
}

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    types: Types<'ctx>,
    mem: GlobalValue<'ctx>,
    io: IO<'ctx>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(
        context: &'ctx Context,
        bf_code: &str,
        mem_size: Option<u32>,
        opt_level: OptimizationLevel,
    ) -> Result<CodeGen<'ctx>, Box<dyn Error>> {
        let module = context.create_module("bf");
        let builder = context.create_builder();
        let execution_engine = module.create_jit_execution_engine(opt_level)?;

        let i8_type = context.i8_type();
        let i32_type = context.i32_type();
        let mem_type = i8_type.array_type(mem_size.unwrap_or(30_000));

        let mem = module.add_global(mem_type, Some(AddressSpace::default()), "mem");
        mem.set_initializer(&mem_type.const_zero());

        let putchar =
            module.add_function("putchar", i32_type.fn_type(&[i32_type.into()], false), None);
        let getchar = module.add_function("getchar", i32_type.fn_type(&[], false), None);

        let new_gen = CodeGen {
            context,
            module,
            builder,
            execution_engine,
            types: Types {
                i8_type,
                i32_type,
                mem_type,
            },
            mem,
            io: IO { getchar, putchar },
        };
        new_gen.add_bf_fn(bf_code)?;

        Ok(new_gen)
    }

    fn add_bf_fn(&self, body: &str) -> Result<(), Box<dyn Error>> {
        let bf_fn_type = self.types.i32_type.fn_type(&[], false);
        let bf_fn = self.module.add_function(MAIN, bf_fn_type, None);
        let basic_block = self.context.append_basic_block(bf_fn, "entry");
        self.builder.position_at_end(basic_block);

        let ptr_alloca = self.builder.build_alloca(self.types.i32_type, "ptr")?;
        self.builder
            .build_store(ptr_alloca, self.types.i32_type.const_zero())?;

        let mut loop_stack = vec![];

        for char in body.chars() {
            match char {
                '>' => self.emit_change_ptr(&ptr_alloca, OpType::Inc),
                '<' => self.emit_change_ptr(&ptr_alloca, OpType::Dec),
                '+' => self.emit_change_val(&ptr_alloca, OpType::Inc),
                '-' => self.emit_change_val(&ptr_alloca, OpType::Dec),
                '.' => self.emit_putchar(&ptr_alloca),
                ',' => self.emit_getchar(&ptr_alloca),
                '[' => self.emit_loop_start(&bf_fn, &mut loop_stack, &ptr_alloca),
                ']' => self.emit_loop_end(&mut loop_stack),
                _ => Ok(()),
            }?
        }

        if !loop_stack.is_empty() {
            return Err("unmatched [".into());
        }

        self.builder
            .build_return(Some(&self.types.i32_type.const_zero()))?;

        Ok(())
    }

    pub fn get_jit_fn(&self) -> Result<JitFunction<BfFunc>, Box<dyn Error>> {
        unsafe { Ok(self.execution_engine.get_function(MAIN)?) }
    }

    fn emit_change_ptr(
        &self,
        &ptr_alloca: &PointerValue,
        op_type: OpType,
    ) -> Result<(), Box<dyn Error>> {
        let ptr = self
            .builder
            .build_load(self.types.i32_type, ptr_alloca, "ptr")?
            .into_int_value();
        let new_ptr = match op_type {
            OpType::Inc => {
                self.builder
                    .build_int_add(ptr, self.types.i32_type.const_int(1, false), "inc")
            }
            OpType::Dec => {
                self.builder
                    .build_int_sub(ptr, self.types.i32_type.const_int(1, false), "dec")
            }
        }?;
        self.builder.build_store(ptr_alloca, new_ptr)?;

        Ok(())
    }

    fn emit_change_val(
        &self,
        &ptr_alloca: &PointerValue,
        op_type: OpType,
    ) -> Result<(), Box<dyn Error>> {
        let i32_type = self.context.i32_type();

        let ptr = self
            .builder
            .build_load(i32_type, ptr_alloca, "ptr")?
            .into_int_value();
        let gep = unsafe {
            self.builder.build_gep(
                self.types.mem_type,
                self.mem.as_pointer_value(),
                &[i32_type.const_zero(), ptr],
                "cell_ptr",
            )
        }?;
        let val = self
            .builder
            .build_load(self.types.i8_type, gep, "val")?
            .into_int_value();

        let new_val = match op_type {
            OpType::Inc => {
                self.builder
                    .build_int_add(val, self.types.i8_type.const_int(1, false), "inc_val")
            }
            OpType::Dec => {
                self.builder
                    .build_int_sub(val, self.types.i8_type.const_int(1, false), "dec_val")
            }
        }?;
        self.builder.build_store(gep, new_val)?;

        Ok(())
    }

    fn emit_putchar(&self, &ptr_alloca: &PointerValue) -> Result<(), Box<dyn Error>> {
        let ptr = self
            .builder
            .build_load(self.types.i32_type, ptr_alloca, "ptr")?
            .into_int_value();
        let gep = unsafe {
            self.builder.build_gep(
                self.types.mem_type,
                self.mem.as_pointer_value(),
                &[self.types.i32_type.const_zero(), ptr],
                "cell_ptr",
            )
        }?;
        let val = self
            .builder
            .build_load(self.types.i8_type, gep, "val")?
            .into_int_value();
        let val_i32 = self
            .builder
            .build_int_z_extend(val, self.types.i32_type, "val_i32")?;

        self.builder
            .build_call(self.io.putchar, &[val_i32.into()], "putchar_call")?
            .try_as_basic_value()
            .left()
            .ok_or("putchar returned void")?;

        Ok(())
    }

    fn emit_getchar(&self, &ptr_alloca: &PointerValue) -> Result<(), Box<dyn Error>> {
        let getchar_result = self
            .builder
            .build_call(self.io.getchar, &[], "getchar_call")?
            .try_as_basic_value()
            .left()
            .ok_or("getchar returned void")?
            .into_int_value();

        let ptr = self
            .builder
            .build_load(self.types.i32_type, ptr_alloca, "ptr")?
            .into_int_value();
        let gep = unsafe {
            self.builder.build_gep(
                self.types.mem_type,
                self.mem.as_pointer_value(),
                &[self.types.i32_type.const_zero(), ptr],
                "cell_ptr",
            )
        }?;

        let ch_trunc =
            self.builder
                .build_int_truncate(getchar_result, self.types.i8_type, "ch_trunc")?;

        self.builder.build_store(gep, ch_trunc)?;

        Ok(())
    }

    fn emit_loop_start(
        &self,
        &cur_fn: &FunctionValue<'ctx>,
        loop_stack: &mut Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,
        &ptr_alloca: &PointerValue,
    ) -> Result<(), Box<dyn Error>> {
        let loop_start = self.context.append_basic_block(cur_fn, "loop_start");
        let loop_body = self.context.append_basic_block(cur_fn, "loop_body");
        let loop_end = self.context.append_basic_block(cur_fn, "loop_end");

        self.builder.build_unconditional_branch(loop_start)?;
        self.builder.position_at_end(loop_start);

        let ptr = self
            .builder
            .build_load(self.types.i32_type, ptr_alloca, "ptr")?
            .into_int_value();
        let gep = unsafe {
            self.builder.build_gep(
                self.types.mem_type,
                self.mem.as_pointer_value(),
                &[self.types.i32_type.const_zero(), ptr],
                "cell_ptr",
            )
        }?;
        let val = self
            .builder
            .build_load(self.types.i8_type, gep, "val")?
            .into_int_value();

        let cond = self.builder.build_int_compare(
            IntPredicate::EQ,
            val,
            self.types.i8_type.const_zero(),
            "cond",
        )?;
        self.builder
            .build_conditional_branch(cond, loop_end, loop_body)?;

        self.builder.position_at_end(loop_body);
        loop_stack.push((loop_start, loop_end));

        Ok(())
    }

    fn emit_loop_end(
        &self,
        loop_stack: &mut Vec<(BasicBlock, BasicBlock)>,
    ) -> Result<(), Box<dyn Error>> {
        let (loop_start, loop_end) = loop_stack.pop().ok_or("unmatched ]")?;
        self.builder.build_unconditional_branch(loop_start)?;
        self.builder.position_at_end(loop_end);

        Ok(())
    }
}

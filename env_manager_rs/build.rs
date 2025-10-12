fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Don't specify out_dir - let tonic use the default OUT_DIR
    // The generated files will be available via include_proto!
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                "../proto/env_manager.v1.proto",
            ],
            &["../proto"],
        )?;
    
    // Emit cargo directives to rerun build script if proto files change
    println!("cargo:rerun-if-changed=../proto/env_manager.v1.proto");
    println!("cargo:rerun-if-changed=../proto");
    
    Ok(())
}


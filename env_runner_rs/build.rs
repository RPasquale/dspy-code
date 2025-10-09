fn main() {
    if let Ok(protoc) = protoc_bin_vendored::protoc_bin_path() {
        std::env::set_var("PROTOC", protoc);
    }

    let proto_root = std::env::var("PROTO_ROOT").unwrap_or_else(|_| "../proto".to_string());
    let runner_proto = format!("{}/runner.proto", proto_root);
    let mesh_proto = format!("{}/mesh.proto", proto_root);
    let proto_files = [runner_proto.as_str(), mesh_proto.as_str()];

    println!("cargo:rerun-if-changed={}/buf.yaml", proto_root);
    println!("cargo:rerun-if-changed=buf.gen.yaml");
    for proto in &proto_files {
        println!("cargo:rerun-if-changed={proto}");
    }

    let mut config = prost_build::Config::new();
    config.bytes([
        ".runner.v1.TaskAssignment.payload",
        ".mesh.v1.SendRequest.payload",
        ".mesh.v1.Received.payload",
    ]);

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_with_config(config, &proto_files, &[proto_root.as_str()])
        .expect("failed to compile protos");
}

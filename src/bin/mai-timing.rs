//! mai-timing CLI binary entry point.

fn main() {
    #[cfg(feature = "cli")]
    {
        if let Err(e) = maghni_flow::cli::run() {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }

    #[cfg(not(feature = "cli"))]
    {
        eprintln!("CLI feature not enabled. Rebuild with --features cli");
        std::process::exit(1);
    }
}

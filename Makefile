.PHONY: build lint test clean

help: ## 📚 Show help for each of the Makefile recipes
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build: ## 🔨 Build the client
	cargo build --workspace

lint: ## 🧹 Linter check
	cargo clippy --all-targets --all-features --workspace -- -D warnings

test: ## 🧪 Run each crate's tests
	cargo test --workspace

clean: ## 🧹 Remove build artifacts
	cargo clean

bench: ## 📊 Run benchmarks
	cargo build --bench db_benchmark && cargo bench --bench db_benchmark

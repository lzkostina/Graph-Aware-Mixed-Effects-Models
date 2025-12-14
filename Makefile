# -------------------------------
# Configuration
# -------------------------------
PYTHON := python
NOTEBOOK := demo.ipynb
RESULTS_DIR := results
EXEC_NB := demo/demo_executed.ipynb
# -------------------------------
# Default target
# -------------------------------
.PHONY: all
all: demo

# -------------------------------
# Run demo notebook
# -------------------------------
.PHONY: demo
demo:
	@echo "Running demo notebook: $(NOTEBOOK)"
	@test -f "$(NOTEBOOK)" || (echo "ERROR: Notebook not found: $(NOTEBOOK)"; exit 2)
	@mkdir -p $(RESULTS_DIR)

	@$(PYTHON) -m jupyter nbconvert \
		--to notebook \
		--execute "$(NOTEBOOK)" \
		--ExecutePreprocessor.timeout=1800 \
		--output "$(notdir $(EXEC_NB))" \
		--output-dir "$(dir $(EXEC_NB))"


# -------------------------------
# Run tests
# -------------------------------
.PHONY: test
test:
	@echo "Running tests..."
	PYTHONPATH=. pytest -q


# -------------------------------
# Clean generated files
# -------------------------------
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -f demo_executed.ipynb
	rm -rf $(RESULTS_DIR)
	rm -rf __pycache__ .pytest_cache

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make demo   Run demo notebook"
	@echo "  make test   Run tests"
	@echo "  make clean  Remove generated files"

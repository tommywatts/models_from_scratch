TESTS=./tests/test*.py

run-tests:
	$(foreach file, $(TESTS), python $(file);)

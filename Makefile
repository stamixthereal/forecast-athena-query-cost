SCRIPTS_DIR = scripts

VENV_DIR = venv

# Set up the virtual environment and install dependencies
.PHONY: local-venv-setup
local-venv-setup:
	python3 -m venv $(VENV_DIR)
	./$(VENV_DIR)/bin/pip install -r requirements.txt

# Grant execution permissions to scripts
.PHONY: grant-execution-permissions
grant-execution-permissions:
	chmod +x $(SCRIPTS_DIR)/*

# Run tests
.PHONY: test
test: grant-execution-permissions
	./$(SCRIPTS_DIR)/run-tests.sh

# Start the application using Docker
.PHONY: start-app-docker
start-app-docker: grant-execution-permissions
	./$(SCRIPTS_DIR)/start-app.sh

# Clean up resources
.PHONY: clean-up-resources
clean-up-resources: grant-execution-permissions
	./$(SCRIPTS_DIR)/clean-up-resources.sh

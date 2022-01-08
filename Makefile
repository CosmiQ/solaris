VENV_DIR = .venv
VENV_PATH = $(VENV_DIR)/bin
PWD = $(shell pwd)

.PHONY: all
all: install_dev_dependencies

# Python Setup
.venv/bin/activate:
	@echo "${bold}Setting up virtualenv...${reset}"
	python3 -m venv ${VENV_DIR}

.PHONY: install_dependencies
install_dependencies: .venv/bin/activate
	@echo "${bold}Installing dependencies...${reset}"
	$(VENV_PATH)/python setup.py install

.PHONY: install_dev_dependencies
install_dev_dependencies: .venv/bin/activate
	@echo "${bold}Installing dependencies...${reset}"
	$(VENV_PATH)/python setup.py develop
	$(VENV_PATH)/pip install -r requirements-test.txt

# Run
.PHONY: test
test: install_dev_dependencies
	$(VENV_PATH)/tox

# CI
.PHONY: lint
lint:
	@echo "${bold}Linting...${reset}"
	${VENV_PATH}/black solaris
	${VENV_PATH}/isort solaris
	${VENV_PATH}/flake8 solaris

.PHONY: lint-ci
lint-ci:
	@echo "${bold}Linting...${reset}"
	${VENV_PATH}/black solaris --check
	${VENV_PATH}/isort solaris --check

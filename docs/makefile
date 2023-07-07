# Define the Python interpreter to use
ENV = C:/ProgramData/Miniconda3/envs/route11
BIB :=
INTERPRETER = $(ENV)/python
PANDOC_OPTIONS = -s --template=eisvogel.latex --listings --number-section
# Define a list of Python scripts to execute
SCRIPTS = report

# Define the default target
all: $(SCRIPTS)


# Define a target to execute all Python scripts
run: all

report: 
	mkdir -p report_output
	python convert_data.py
	pandoc temp.md $(PANDOC_OPTIONS) -o report_output/CPT_correlations.pdf
	rm -f temp.md
	cmd /c start report_output/CPT_correlations.pdf


clean:
	rm -f *.html
	rm -f *.md
	rm -f *.pdf
	rm -f *.json
	rm -f *.pyc
	rm -f *.log

# Define a target for help
help:
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@echo "  all         - execute all Python scripts"
	@echo "  run         - execute all Python scripts"
	@echo " -B"          - to force rerun the script"
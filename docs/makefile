# Define the Python interpreter to use
ENV = C:/ProgramData/Miniconda3/envs/route11
INTERPRETER = $(ENV)/python
# PANDOC_OPTIONS = -s --template=eisvogel.latex --listings --number-section --filter=pandoc-citeproc
PANDOC_OPTIONS = -s  --number-sections --citeproc --template eisvogel
# PANDOC_OPTIONS = -s  --citeproc --number-sections --listing
# PANDOC_OPTIONS = -s  --number-section --citeproc 
# PANDOC_OPTIONS = -s --listings --number-section --citeproc --template=eisvogel.latex_ --csl=cambridge-university-press-author-date.csl
# Define a list of Python scripts to execute
SCRIPTS = report
MD_FILES = $(wildcard *.md)
TXT_FILES = $(sort $(subst .md,.txt, $(MD_FILES)))
# Define the default target

$(info $(MD_FILES))
$(info $(TXT_FILES))

all: report.pdf


# Define a target to execute all Python scripts
run: all

report.pdf: $(TXT_FILES)
	mkdir -p report_output
	pandoc $^ $(PANDOC_OPTIONS) -o $@
	rm -f $(TXT_FILES)
	cmd /c start $@

%.txt: %.md
	python convert_data.py $< $@

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

# UnitCPT

UnitCPT is a Python package built on top of the `Plotly Dash` framework, designed to provide functionality for processing CPT data for geological interpretation and offshore foundation design.



## Installation

 ```shell
 conda create -n unitcpt python=3.9
 conda activate unitcpt

 ```

## Create and delete projects

This function is added to enable create and delete project easily with an interface

## How to communicate between different pages

1. designed a data structure in `app.py`, when calling the layout, this needs to be function, such that it will get updated when switching back to the original page.
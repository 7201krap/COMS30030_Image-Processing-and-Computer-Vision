#!/bin/bash
activate(){
	python3 -m venv ./myenv
	. ./myenv/bin/activate
	pip3 install --no-cache-dir --upgrade pip
	pip3 install --no-cache-dir scikit-build
	pip3 install --no-cache-dir numpy  
	pip3 install --no-cache-dir opencv-python
}
activate

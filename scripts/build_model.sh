#!/usr/bin/env bash

bentoml build --version 1.0 -f models/bentofile.yaml
bentoml containerize -t iris_classifier:1.0 --opt platform=linux/amd64 iris_classifier:latest
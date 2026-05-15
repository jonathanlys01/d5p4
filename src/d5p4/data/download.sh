#!/bin/bash

REPO_ID="HuggingFaceFW/fineweb-edu"
PATH_TO_LOCAL_DIR=$1

huggingface-cli download $REPO_ID --local-dir $PATH_TO_LOCAL_DIR --repo-type dataset --include "sample/10BT/*"
chmod -R 777 $PATH_TO_LOCAL_DIR
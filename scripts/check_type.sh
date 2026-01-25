#!/bin/bash

if [ "$CONDA_DEFAULT_ENV" != "videoedit" ]; then
    echo "当前环境: $CONDA_DEFAULT_ENV"
    echo "正在切换到videoedit环境..."
    eval "$(conda shell.bash hook)"
    conda activate videoedit
    echo "✓ 已切换到videoedit环境"
else
    echo "✓ 已在videoedit环境中"
fi

mypy src --ignore-missing-imports
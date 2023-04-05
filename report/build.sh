#!/bin/bash
docker build -t thesis . --progress=plain && docker run --rm --entrypoint cat thesis /app/title-page-ai.pdf > ../title-page-ai.pdf && docker run --rm --entrypoint cat thesis /app/neurips_2022.pdf > ../neurips_2022.pdf


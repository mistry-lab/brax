#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

NEWEST_RUN=$(find $1 -name params -exec stat -c '%Z %n' {} + | sort -r | head -n 1 | awk '{print $2}')
ls -d $NEWEST_RUN/*/ | xargs stat -c '%Z %n' | sort -r | head -n 1 | awk '{print $2}'

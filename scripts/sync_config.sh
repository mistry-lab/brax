#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/.env

CMD="scp -r alexaldermanwebb@${3-$ADDRESS}:${1-./brax/brax/scripts/cfg} ${2-$SCRIPT_DIR}"
if ! [ -z $4 ]; then
	sshpass -p $4 $CMD
else
	$CMD
fi

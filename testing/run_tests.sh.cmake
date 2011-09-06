#!/usr/bin/env bash
# Copyright 2011, Tokyo Institute of Technology.
# All rights reserved.
#
# This file is distributed under the license described in
# LICENSE.txt.
#
# Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

# TODO:
# - translation test
# - execution test

###############################################################
WD=$PWD/test_output
FLAG_KEEP_OUTPUT=1
###############################################################
set -u
set -e
TIMESTAMP=$(date +%m-%d-%Y_%H-%M-%S)
LOGFILE=$PWD/run_tests.log.$TIMESTAMP
WD=$WD/$TIMESTAMP
ORIGINAL_DIRECTORY=$PWD
mkdir -p $WD
cd $WD


NUM_SUCCESS=0
NUM_FAIL=0

function finish()
{
	echo "All tests completed. #SUCCESS: $NUM_SUCCESS, #FAIL: $NUM_FAIL."
	cd $ORIGINAL_DIRECTORY
	if [ $FLAG_KEEP_OUTPUT -eq 0 ]; then
		rm -rf $WD
	fi
	if [ $NUM_FAIL -eq 0 ]; then
		exit 0
	else
		exit 1
	fi
}

{
	# find tests
	TESTS=$(find ${CMAKE_SOURCE_DIR}/testing/tests -name 'test_*.c')
	NUM_TESTS=$(echo $TESTS | wc -w)
	echo Running $NUM_TESTS tests
	PHYSISC=${CMAKE_BINARY_DIR}/translator/physisc
	TARGETS=$($PHYSISC --list-targets)
	for TARGET in $TARGETS; do
		for TEST in $TESTS; do
			echo Translating $TEST to $TARGET code
			if $PHYSISC --$TARGET -I${CMAKE_SOURCE_DIR}/include $TEST \
				> physisc.$TEST.$TARGET.log 2>&1; then
				echo "\[TRANSLATION\] $TEST SUCCESS"
				NUM_SUCCESS=$(($NUM_SUCCESS + 1))
			else
				echo "\[TRANSLATION\] $TEST FAIL"
				NUM_FAIL=$(($NUM_FAIL + 1))
			# Terminates immediately when failed
				finish
			fi
		done
	done
	finish
} >> $LOGFILE 2>&1

# Local Variables:
# mode: sh-mode
# End:


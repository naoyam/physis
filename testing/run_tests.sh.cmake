#!/usr/bin/env bash
# Copyright 2011, Tokyo Institute of Technology.
# All rights reserved.
#
# This file is distributed under the license described in
# LICENSE.txt.
#
# Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

# TODO:
# - compilation
# - linking
# - execution

###############################################################
WD=$PWD/test_output
FLAG_KEEP_OUTPUT=1
if [ "x$CFLAGS" = "x" ]; then
	CFLAGS=""
fi
###############################################################
set -u
#set -e
TIMESTAMP=$(date +%m-%d-%Y_%H-%M-%S)
LOGFILE=$PWD/run_tests.log.$TIMESTAMP
WD=$WD/$TIMESTAMP
ORIGINAL_DIRECTORY=$PWD
mkdir -p $WD
cd $WD

NUM_SUCCESS_TRANS=0
NUM_FAIL_TRANS=0
NUM_SUCCESS_COMPILE=0
NUM_FAIL_COMPILE=0
NUM_SUCCESS_EXEC=0
NUM_FAIL_EXEC=0

function finish()
{
	echo "All tests completed."
	echo  "[TRANSLATE] #SUCCESS: $NUM_SUCCESS_TRANS, #FAIL: $NUM_FAIL_TRANS."
	echo  "[COMPILE]   #SUCCESS: $NUM_SUCCESS_COMPILE, #FAIL: $NUM_FAIL_COMPILE."
	echo  "[EXECUTE]   #SUCCESS: $NUM_SUCCESS_EXEC, #FAIL: $NUM_FAIL_EXEC."
	cd $ORIGINAL_DIRECTORY
	if [ $FLAG_KEEP_OUTPUT -eq 0 ]; then
		rm -rf $WD
	fi
	if [ $NUM_FAIL_TRANS -eq 0 -a $NUM_FAIL_COMPILE -eq 0 -a \
		$NUM_FAIL_EXEC -eq 0 ]; then
		exit 0
	else
		exit 1
	fi
}

function compile()
{
	src=$1
	target=$2
	echo compilation of $src for $target
	src_file_base=$(basename $src .c).$target
	if [ "${MPI_ENABLED}" = "TRUE" ]; then
		MPI_C_FLAGS="-pthread"
		for mpiinc in $(echo "${MPI_INCLUDE_PATH}" | sed 's/;/ /g'); do
			MPI_C_FLAGS+=" -I$mpiinc"
		done
	fi
	LDFLAGS=-L${CMAKE_BINARY_DIR}/runtime
	case $target in
		ref)
			src_file="$src_file_base".c
			cc -c $src_file -I${CMAKE_SOURCE_DIR}/include $CFLAGS &&
			c++ "$src_file_base".o -lphysis_rt_ref $LDFLAGS -o "$src_file_base".exe
			;;
		cuda)
			if [ "${CUDA_ENABLED}" != "TRUE" ]; then
				echo "[COMPILE] Skipping CUDA compilation (not supported)"
				return 0
			fi
			src_file="$src_file_base".cu
			nvcc -c $src_file -I${CMAKE_SOURCE_DIR}/include -Xcompiler $CFLAGS
			;;
		mpi)
			if [ "${MPI_ENABLED}" != "TRUE" ]; then
				echo "[COMPILE] Skipping MPI compilation (not supported)"
				return 0
			fi
			src_file="$src_file_base".c			
			cc -c $src_file -I${CMAKE_SOURCE_DIR}/include $MPI_C_FLAGS $CFLAGS
			;;
		mpi-cuda)
			if [ "${MPI_ENABLED}" != "TRUE" -o "${CUDA_ENABLED}" != "TRUE" ]; then
				echo "[COMPILE] Skipping MPI-CUDA compilation (not supported)"
				return 0
			fi
			src_file="$src_file_base".cu			
			nvcc -c $src_file -I${CMAKE_SOURCE_DIR}/include \
				$MPI_C_FLAGS -Xcompiler $CFLAGS 
			;;
		*)
			echo "Error: Unsupported target"
			finish
			;;
	esac
}

{
	# find tests
	TESTS=$(find ${CMAKE_SOURCE_DIR}/testing/tests -name 'test_*.c')
	NUM_TESTS=$(echo $TESTS | wc -w)
	echo "Testing $NUM_TESTS test code(s)"
	PHYSISC=${CMAKE_BINARY_DIR}/translator/physisc
	TARGETS=$($PHYSISC --list-targets)
	for TARGET in $TARGETS; do
		for TEST in $TESTS; do
			SHORTNAME=$(basename $TEST)
			echo "[TRANSLATE] Processing $SHORTNAME for $TARGET target"
			if $PHYSISC --$TARGET -I${CMAKE_SOURCE_DIR}/include $TEST \
				> $(basename $TEST).$TARGET.log 2>&1; then
				echo "[TRANSLATE] SUCCESS"
				NUM_SUCCESS_TRANS=$(($NUM_SUCCESS_TRANS + 1))
			else
				echo "[TRANSLATE] FAIL"
				NUM_FAIL_TRANS=$(($NUM_FAIL_TRANS + 1))
			    # Terminates immediately when failed
				finish
			fi
			echo "[COMPILE] Processing $SHORTNAME for $TARGET target"
			if compile $SHORTNAME $TARGET; then
				echo "[COMPILE] SUCCESS"
				NUM_SUCCESS_COMPILE=$(($NUM_SUCCESS_COMPILE + 1))
			else
				echo "[COMPILE] FAIL"
				NUM_FAIL_COMPILE=$(($NUM_FAIL_COMPILE + 1))
				# Terminates immediately when failed
				finish
			fi
		done
	done
	finish
} 2>&1 | tee $LOGFILE

# Local Variables:
# mode: sh-mode
# End:


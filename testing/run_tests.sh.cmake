#!/usr/bin/env bash
# Copyright 2011, Tokyo Institute of Technology.
# All rights reserved.
#
# This file is distributed under the license described in
# LICENSE.txt.
#
# Author: Naoya Maruyama (naoya@matsulab.is.titech.ac.jp)

#
# For usage, see the help message by executing this script with option --help.
#

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
NUM_SUCCESS_EXECUTE=0
NUM_FAIL_EXECUTE=0
FAILED_TESTS=""

PHYSISC=${CMAKE_BINARY_DIR}/translator/physisc
MPIRUN="MPIRUN_UNSET"

function fail()
{
	test=$1
	trg=$2
	stage=$3
	FAILED_TESTS+="$test/$trg/$stage "
}

function finish()
{
	echo "All tests completed."
	echo  "[TRANSLATE] #SUCCESS: $NUM_SUCCESS_TRANS, #FAIL: $NUM_FAIL_TRANS."
	echo  "[COMPILE]   #SUCCESS: $NUM_SUCCESS_COMPILE, #FAIL: $NUM_FAIL_COMPILE."
	echo  "[EXECUTE]   #SUCCESS: $NUM_SUCCESS_EXECUTE, #FAIL: $NUM_FAIL_EXECUTE."
	if [ "x" != "x$FAILED_TESTS" ]; then echo  "Failed tests: $FAILED_TESTS"; fi
	cd $ORIGINAL_DIRECTORY
	if [ $FLAG_KEEP_OUTPUT -eq 0 ]; then
		rm -rf $WD
	fi
	if [ $NUM_FAIL_TRANS -eq 0 -a $NUM_FAIL_COMPILE -eq 0 -a \
		$NUM_FAIL_EXECUTE -eq 0 ]; then
		exit 0
	else
		exit 1
	fi
}

function compile()
{
	src=$1
	target=$2
	src_file_base=$(basename $src .c).$target
	if [ "${MPI_ENABLED}" = "TRUE" ]; then
		MPI_CFLAGS="-pthread"
		for mpiinc in $(echo "${MPI_INCLUDE_PATH}" | sed 's/;/ /g'); do
			MPI_CFLAGS+=" -I$mpiinc"
		done
	fi
	LDFLAGS=-L${CMAKE_BINARY_DIR}/runtime
	NVCC_CFLAGS="-arch sm_20"
	CUDA_LDFLAGS="-lcudart -L$(dirname ${CUDA_LIBRARIES})"
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
			nvcc -c $src_file -I${CMAKE_SOURCE_DIR}/include $NVCC_CFLAGS $CFLAGS &&
			nvcc "$src_file_base".o -lphysis_rt_cuda $LDFLAGS -o "$src_file_base".exe
			;;
		mpi)
			if [ "${MPI_ENABLED}" != "TRUE" ]; then
				echo "[COMPILE] Skipping MPI compilation (not supported)"
				return 0
			fi
			src_file="$src_file_base".c			
			cc -c $src_file -I${CMAKE_SOURCE_DIR}/include $MPI_CFLAGS $CFLAGS &&
			mpic++ "$src_file_base".o -lphysis_rt_mpi $LDFLAGS -o "$src_file_base".exe
			;;
		mpi-cuda)
			if [ "${MPI_ENABLED}" != "TRUE" -o "${CUDA_ENABLED}" != "TRUE" ]; then
				echo "[COMPILE] Skipping MPI-CUDA compilation (not supported)"
				return 0
			fi
			src_file="$src_file_base".cu
			MPI_CUDA_CFLAGS=""
			for f in $CFLAGS $MPI_CFLAGS; do
				if echo $f | grep '^-I' > /dev/null; then
					MPI_CUDA_CFLAGS+="$f "
				else
					MPI_CUDA_CFLAGS+="-Xcompiler $f "
				fi
			done
			nvcc -c $src_file -I${CMAKE_SOURCE_DIR}/include \
				$NVCC_CFLAGS $MPI_CUDA_CFLAGS &&
			mpic++ "$src_file_base".o -lphysis_rt_mpi_cuda $LDFLAGS $CUDA_LDFLAGS \
			-o "$src_file_base".exe
			;;
		*)
			echo "ERROR! Unsupported target"
			finish
			;;
	esac
}

# TODO
function execute()
{
	src=$1
	target=$2
	src_file_base=$(basename $src .c).$target
	exename=$src_file_base.exe	
	echo "[EXECUTE] Executing $exename"
	case $target in
		ref)
			./$exename
			;;
		cuda)
			./$exename
			;;
		mpi)
			$MPIRUN ./$exename 
			;;
		mpi-cuda)
			$MPIRUN ./$exename 
			;;
		*)
			echo "ERROR! Unsupported target"
			finish
			;;
	esac
}

function print_usage()
{
	echo "USAGE"
	echo -e "\trun_tests.sh [options]"
	echo ""
	echo "OPTIONS"
	echo -e "\t-t, --targets"
	echo -e "\t\tSet the test targets. Supported targets: $($PHYSISC --list-targets)."
	echo -e "\t-s, --source"
	echo -e "\t\tSet the test source files."
	echo -e "\t--translate"
	echo -e "\t\tTest only translation and its dependent tasks."
	echo -e "\t--compile"
	echo -e "\t\tTest only compilation and its dependent tasks."
	echo -e "\t--execute"
	echo -e "\t\tTest only execution and its dependent tasks."
	echo -e "\t-m, --mpirun"
	echo -e "\t\tThe mpirun command for testing MPI-based runs. Include necessary options like -np within a quoted string."
}

{

	TARGETS=""
	STAGE="ALL"
	
	# find tests
	TESTS=$(find ${CMAKE_SOURCE_DIR}/testing/tests -name 'test_*.c'|sort -n)

	TEMP=$(getopt -o ht:s:m: --long help,targets:,source:,translate,compile,execute,mpirun -- "$@")
	if [ $? != 0 ]; then
		echo "ERROR! Invalid options: $@";
		print_usage
		exit 1;
	fi
	
	eval set -- "$TEMP"
	while true; do
		case "$1" in
			-t|--targets)
				TARGETS=$2
				shift 2
				;;
			-s|--source)
				SRC=$2
				TMP=""
				for i in $TESTS; do
					for j in $SRC; do
						if echo $i | grep --silent $j; then
							TMP+="$i "
						fi
					done
				done
				TESTS=$TMP
				shift 2
				;;
			--translate)
				STAGE="TRANSLATE"
				shift
				;;
			--compile)
				STAGE="COMPILE"
				shift
				;;
			--execute)
				STAGE="EXECUTE"
				shift
				;;
			-m|--mpi-run)
				MPIRUN=$2
				shift 2
				;;
			-h|--help)
				print_usage
				exit 0
				shift
				;;
			--)
				shift
				break
				;;
			*) 
				echo "ERROR! Invalid option: $1"
				print_usage
				exit 1
				;;
		esac
	done
	
	# Test all targets by default
	if [ "x$TARGETS" = "x" ]; then TARGETS=$($PHYSISC --list-targets); fi
	
	
	echo "Test sources: $(for i in $TESTS; do basename $i; done | xargs)"
	echo "Targets: $TARGETS"
	
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
				fail $SHORTNAME $TARGET translate
				continue
			fi
			if [ "$STAGE" = "TRANSLATE" ]; then continue; fi
			echo "[COMPILE] Processing $SHORTNAME for $TARGET target"
			if compile $SHORTNAME $TARGET; then
				echo "[COMPILE] SUCCESS"
				NUM_SUCCESS_COMPILE=$(($NUM_SUCCESS_COMPILE + 1))
			else
				echo "[COMPILE] FAIL"
				NUM_FAIL_COMPILE=$(($NUM_FAIL_COMPILE + 1))
				fail $SHORTNAME $TARGET compile
				continue				
			fi
			if [ "$STAGE" = "COMPILE" ]; then continue; fi
			if execute $SHORTNAME $TARGET; then
				echo "[EXECUTE] SUCCESS"
				NUM_SUCCESS_EXECUTE=$(($NUM_SUCCESS_EXECUTE + 1))
			else
				echo "[EXECUTE] FAIL"
				NUM_FAIL_EXECUTE=$(($NUM_FAIL_EXECUTE + 1))
				fail $SHORTNAME $TARGET execute
				continue
			fi
		done
	done
	finish
} 2>&1 | tee $LOGFILE

# Local Variables:
# mode: sh-mode
# End:


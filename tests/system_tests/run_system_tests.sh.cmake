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
DIE_IMMEDIATELY=0
###############################################################
set -u
#set -e
TIMESTAMP=$(date +%m-%d-%Y_%H-%M-%S)
LOGFILE=$PWD/$(basename $0 .sh).log.$TIMESTAMP
WD=$WD/$TIMESTAMP
ORIGINAL_DIRECTORY=$PWD
mkdir -p $WD
ORIGINAL_WD=$(pwd)
cd $WD

NUM_SUCCESS_TRANS=0
NUM_FAIL_TRANS=0
NUM_SUCCESS_COMPILE=0
NUM_FAIL_COMPILE=0
NUM_SUCCESS_EXECUTE=0
NUM_FAIL_EXECUTE=0
NUM_WARNING=0
FAILED_TESTS=""

PHYSISC=${CMAKE_BINARY_DIR}/translator/physisc
MPIRUN=mpirun
MPI_PROC_DIM="1,1x1,1x1x1" # use only 1 process by default; can be controlled by the -np option
MPI_MACHINEFILE=""

function print_error()
{
    echo "ERROR!: $1" >&2
}

function warn()
{
    echo "WARNING!: $1" >&2
    NUM_WARNING=$(($NUM_WARNING + 1))
}

function fail()
{
    local msg=$1
    shift
    while [ $# -gt 0 ]; do
	msg+="/$1"
	shift
    done
    FAILED_TESTS+="$msg "
    if [ $DIE_IMMEDIATELY -eq 1 ]; then
	exit_error
    fi
}

function finish()
{
    echo  "[TRANSLATE] #SUCCESS: $NUM_SUCCESS_TRANS, #FAIL: $NUM_FAIL_TRANS."
    echo  "[COMPILE]   #SUCCESS: $NUM_SUCCESS_COMPILE, #FAIL: $NUM_FAIL_COMPILE."
    echo  "[EXECUTE]   #SUCCESS: $NUM_SUCCESS_EXECUTE, #FAIL: $NUM_FAIL_EXECUTE."
    if [ "x" != "x$FAILED_TESTS" ]; then echo  "Failed tests: $FAILED_TESTS"; fi
    if [ $NUM_WARNING -gt 0 ]; then
	echo "$NUM_WARNING warning(s)"
    fi
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

function exit_success()
{
    echo "All tests completed."
    finish
}

function exit_error()
{
    if [ $# -gt 0 ]; then
	print_error $1
    fi
    echo "Testing stopped (some tests are not done)."
    finish
}

function abs_path()
{
    local path=$1
    if [ $(expr substr $path 1 1) = '/' ]; then
	echo $path
    else
	echo $ORIGINAL_WD/$path
    fi
}

function generate_empty_translation_configuration()
{
    echo -n "" > config.empty
    echo config.empty
}

function generate_translation_configurations_ref()
{
    if [ $# -gt 0 ]; then
	echo $1
    else
	generate_empty_translation_configuration
    fi
}

function generate_translation_configurations_cuda()
{
    local configs=""    
    if [ $# -gt 0 ]; then
	configs=$*
    else
	configs=config.empty
	echo -n "" > config.empty
    fi
    local pre_calc='true false'
    local bsize="64,4,1 32,8,1"
    local new_configs=""
    local idx=0
    for i in $pre_calc; do
	for j in $bsize; do
	    for k in $configs; do
		local c=config.cuda.$idx
		idx=$(($idx + 1))
		cat $k > $c
		echo "CUDA_PRE_CALC_GRID_ADDRESS = $i" >> $c
		echo "CUDA_BLOCK_SIZE = {$j}" >> $c
		new_configs="$new_configs $c"
	    done
	done
    done
    echo $new_configs
}

function generate_translation_configurations_mpi()
{
    if [ $# -gt 0 ]; then
	echo $1
    else
	generate_empty_translation_configuration
    fi
}

function generate_translation_configurations_mpi_cuda()
{
    local configs=""    
    if [ $# -gt 0 ]; then
	configs=$*
    else
	configs=config.empty
	echo -n "" > config.empty
    fi
    configs=$(generate_translation_configurations_cuda "$configs")
    local overlap='false true'
    local multistream='false true'
    local new_configs=""
    local idx=0
    for i in $overlap; do
	for j in $multistream; do
	    for k in $configs; do
		# skip configurations with not all options enabled
		if [ \( $i = 'true' -a $j = 'false' \)  \
			-o \( $i = 'false' -a $j = 'true' \) ]; then
		    continue;
		fi
		local c=config.mpi-cuda.$idx
		idx=$(($idx + 1))
		cat $k > $c
		echo "MPI_OVERLAP = $i" >> $c
		echo "MULTISTREAM_BOUNDARY = $j" >> $c
		new_configs="$new_configs $c"
	    done
	done
    done
    echo $new_configs
}

function generate_translation_configurations()
{
    target=$1
    case $target in
	ref)
	    generate_translation_configurations_ref
	    ;;
	cuda)
	    generate_translation_configurations_cuda
	    ;;
	mpi)
	    generate_translation_configurations_mpi
	    ;;
	mpi-cuda)
	    generate_translation_configurations_mpi_cuda
	    ;;
	*)
	    generate_empty_compile_configuration
	    ;;
    esac
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
    CUDA_LDFLAGS="-lcudart -L${CUDA_RT_DIR}"
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
	    nvcc -m64 -c $src_file -I${CMAKE_SOURCE_DIR}/include $NVCC_CFLAGS $CFLAGS &&
	    nvcc -m64 "$src_file_base".o -lphysis_rt_cuda $LDFLAGS \
		'${CUDA_CUT_LIBRARIES}' -o "$src_file_base".exe
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
	    nvcc -m64 -c $src_file -I${CMAKE_SOURCE_DIR}/include \
		$NVCC_CFLAGS $MPI_CUDA_CFLAGS &&
	    mpic++ "$src_file_base".o -lphysis_rt_mpi_cuda $LDFLAGS $CUDA_LDFLAGS \
		'${CUDA_CUT_LIBRARIES}' -o "$src_file_base".exe
	    ;;
	*)
	    exit_error "Unsupported target"
	    ;;
    esac

    sync
}

function get_reference_exe_name()
{
    local src_name=$(basename $1 .c)
    local target=$2
    case $target in
	mpi)
	    target=ref
	    ;;
	mpi-cuda)
	    target=cuda
	    ;;
    esac
    local ref_exe=${CMAKE_CURRENT_BINARY_DIR}/tests/$src_name.manual.$target.exe
    echo $ref_exe
}

function execute_reference()
{
    local src_name=$(basename $1 .c)
    local target=$2
    local ref_exe=$(get_reference_exe_name $1 $2)
    local ref_out=$ref_exe.out    
    # Do nothing if no reference implementation is found.
    if [ ! -x $ref_exe ]; then
	echo "[EXECUTE] No reference implementation found." >&2
	# Check if other implementation variants exist. If true,
	# warn about the lack of an implementation for this target
	if ls ${CMAKE_CURRENT_BINARY_DIR}/tests/$src_name.manual.*.exe > \
	    /dev/null 2>&1 ; then
	    warn "Missing reference implementation for $target?"
	fi
	return 1
    fi
    if [ $ref_exe -ot $ref_out ]; then
	echo "[EXECUTE] Previous output found." >&2
    else 
	echo "[EXECUTE] Executing reference implementation ($ref_exe)." >&2
	$ref_exe > $ref_out
    fi
    return 0
}

function do_mpirun()
{
    local proc_dim_list=$1
    shift
    local dim=$1
    shift
    local mfile=$1
    shift
    local mfile_option="-machinefile $mfile"
    if [ "x$mfile" = "x" ]; then
	mfile_option=""
    fi
    local proc_dim=$(echo $proc_dim_list | cut -d, -f$dim)
    local np=$(($(echo $proc_dim | sed 's/x/*/g')))
    echo "[EXECUTE] Run mpi as \"$MPIRUN -np $np $mfile_option $* --physis-proc $proc_dim\"" >&2
    $MPIRUN -np $np $mfile_option $* --physis-proc $proc_dim
}

function execute()
{
    local target=$2
    local exename=$(basename $1 .c).$target.exe
    echo  "[EXECUTE] Executing $exename"
    case $target in
	ref)
	    ./$exename > $exename.out
	    ;;
	cuda)
	    ./$exename > $exename.out
	    ;;
	mpi)
	    do_mpirun $3 $4  "$MPI_MACHINEFILE" ./$exename > $exename.out
	    ;;
	mpi-cuda)
	    do_mpirun $3 $4 "$MPI_MACHINEFILE" ./$exename > $exename.out	    
	    ;;
	*)
	    exit_error "Unsupported target: $2"
	    ;;
    esac
    if [ $? -ne 0 ]; then
	return 1
    fi
    execute_reference $1 $2
    local ref_output=$(get_reference_exe_name $1 $2).out
    if [ -f "$ref_output" ]; then
	echo "[EXECUTE] Reference output found. Validating output..."
	if ! diff $ref_output $exename.out > $exename.out.diff ; then
	    print_error "Invalid output. Diff saved at: $(pwd)/$exename.out.diff"
	    return 1
	else
	    rm $exename.out.diff
	    echo "[EXECUTE] Successfully validated."
	fi
    fi
}

function print_usage()
{
    echo "USAGE"
    echo -e "\trun_tests.sh [options]"
    echo ""
    echo "OPTIONS"
    echo -e "\t-t, --targets <targets>"
    echo -e "\t\tSet the test targets. Supported targets: $($PHYSISC --list-targets)."
    echo -e "\t-s, --source <source-names>"
    echo -e "\t\tSet the test source files."
    echo -e "\t--translate"
    echo -e "\t\tTest only translation and its dependent tasks."
    echo -e "\t--compile"
    echo -e "\t\tTest only compilation and its dependent tasks."
    echo -e "\t--execute"
    echo -e "\t\tTest only execution and its dependent tasks."
    echo -e "\t-m, --mpirun"
    echo -e "\t\tThe mpirun command for testing MPI-based runs."
    echo -e "\t--proc-dim <proc-dim-list>"
    echo -e "\t\tProcess dimension. E.g., to run 16 processes, specify this \n\t\toption like '16,4x4,1x4x4'. This way, 16 processes are mapped\n\t\tto the overall problem domain with the decomposition for\n\t\th dimensionality. Multiple values can be passed with quotations."
    echo -e "\t--machinefile <file-path>"
    echo -e "\t\tThe MPI machinefile."
}

function get_test_cases()
{
    local test_names=""
    if [ $# -eq 0 ]; then
	test_names="test_1 test_2 test_3 test_4 test_5 test_6 test_7"
    else
	test_names="$*"
    fi
    local tests2=""
    for t in $test_names; do
	tests2+="${CMAKE_CURRENT_SOURCE_DIR}/test_cases/$t.c "
    done
    echo $tests2
}

{
    TARGETS=""
    STAGE="ALL"

    TESTS=$(get_test_cases)
    
    set +e
    TESTS=$(for t in $TESTS; do echo $t | grep -v 'test_.*\.manual\.'; done)
    set -e
    TEMP=$(getopt -o ht:s:m:q --long help,targets:,source:,translate,compile,execute,mpirun,machinefile:,proc-dim:,quit -- "$@")
    if [ $? != 0 ]; then
	print_error "Invalid options: $@"
	print_usage
	exit_error
    fi
    
    eval set -- "$TEMP"
    while true; do
	case "$1" in
	    -t|--targets)
		TARGETS=$2
		shift 2
		;;
	    -s|--source)
		TESTS=$(get_test_cases $2)
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
	    --proc-dim)
		MPI_PROC_DIM=$2
		shift 2
		;;
	    --machinefile)
		MPI_MACHINEFILE=$(abs_path $2)
		shift 2
		;;
	    -q|--quit)
		DIE_IMMEDIATELY=1
		shift
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
		print_error "Invalid option: $1"
		print_usage
		exit_error
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
	    DESC=$(grep -o '\WTEST: .*$' $TEST | sed 's/\WTEST: \(.*\)$/\1/')
	    DIM=$(grep -o '\WDIM: .*$' $TEST | sed 's/\WDIM: \(.*\)$/\1/')
	    for CONFIG in $(generate_translation_configurations $TARGET); do
		echo "Testing with $SHORTNAME ($DESC)"
		echo "Configuration ($CONFIG):"
		echo "Dimension: $DIM"
		cat $CONFIG
		echo "[TRANSLATE] Processing $SHORTNAME for $TARGET target"
		if $PHYSISC --$TARGET -I${CMAKE_SOURCE_DIR}/include --config $CONFIG \
		    $TEST > $(basename $TEST).$TARGET.log 2>&1; then
		    echo "[TRANSLATE] SUCCESS"
		    NUM_SUCCESS_TRANS=$(($NUM_SUCCESS_TRANS + 1))
		else
		    echo "[TRANSLATE] FAIL"
		    NUM_FAIL_TRANS=$(($NUM_FAIL_TRANS + 1))
		    fail $SHORTNAME $TARGET translate $CONFIG
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
		    fail $SHORTNAME $TARGET compile $CONFIG
		    continue				
		fi
		if [ "$STAGE" = "COMPILE" ]; then continue; fi
		np_target=1
		if [ "$TARGET" = "mpi" -o "$TARGET" = "mpi-cuda" ]; then
		    np_target=$MPI_PROC_DIM
		    echo "[EXECUTE] Trying with process configurations: $np_target"
                    # Make sure compiled binaries are synched to other nodes
		    # This is a temporary workaround for the Raccoon cluster.
		    sleep 30
		fi
		for np in $np_target; do
		    if execute $SHORTNAME $TARGET $np $DIM; then
			echo "[EXECUTE] SUCCESS"
			NUM_SUCCESS_EXECUTE=$(($NUM_SUCCESS_EXECUTE + 1))
		    else
			echo "[EXECUTE] FAIL"
			NUM_FAIL_EXECUTE=$(($NUM_FAIL_EXECUTE + 1))
			fail $SHORTNAME $TARGET execute $CONFIG $np
			continue
		    fi
		done
	    done
	done
    done
    finish
} 2>&1 | tee $LOGFILE

# Local Variables:
# mode: sh-mode
# End:


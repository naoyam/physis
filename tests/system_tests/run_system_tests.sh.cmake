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
WD_BASE=$PWD/test_output
FLAG_KEEP_OUTPUT=1
if [ "x$CFLAGS" = "x" ]; then
    CFLAGS=""
fi
CFLAGS="${CFLAGS} -g -Wno-unuused-variable"
DIE_IMMEDIATELY=0
###############################################################
set -u
#set -e
TIMESTAMP=$(date +%m-%d-%Y_%H-%M-%S)
LOGFILE=$PWD/$(basename $0 .sh).log.$TIMESTAMP
WD=$WD_BASE/$TIMESTAMP
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
PHYSIS_NLP=1

EXECUTE_WITH_VALGRIND=0

PRIORITY=1
CONFIG_ARG=""

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

function print_results_short()
{
    echo  -n "[TRANSLATE] #SUCCESS: $NUM_SUCCESS_TRANS, #FAIL: $NUM_FAIL_TRANS"
    echo  -n ", [COMPILE] #SUCCESS: $NUM_SUCCESS_COMPILE, #FAIL: $NUM_FAIL_COMPILE"
    echo  ", [EXECUTE] #SUCCESS: $NUM_SUCCESS_EXECUTE, #FAIL: $NUM_FAIL_EXECUTE."
}

function print_results()
{
    echo  "[TRANSLATE] #SUCCESS: $NUM_SUCCESS_TRANS, #FAIL: $NUM_FAIL_TRANS."
    echo  "[COMPILE]   #SUCCESS: $NUM_SUCCESS_COMPILE, #FAIL: $NUM_FAIL_COMPILE."
    echo  "[EXECUTE]   #SUCCESS: $NUM_SUCCESS_EXECUTE, #FAIL: $NUM_FAIL_EXECUTE."
    if [ "x" != "x$FAILED_TESTS" ]; then echo  "Failed tests: $FAILED_TESTS"; fi
    if [ $NUM_WARNING -gt 0 ]; then
		echo "$NUM_WARNING warning(s)"
    fi
}

function finish()
{
	print_results
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

function clear_output()
{
	rm -f $ORIGINAL_WD/$(basename $0 .sh).log.*
	rm -rf $WD_BASE
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
    local new_configs=""
    local idx=0
    local c=config.ref.$idx
	touch $c
    new_configs="$new_configs $c"
	idx=$(($idx + 1))

    c=config.ref.$idx
    echo "OPT_KERNEL_INLINING = true" > $c
    new_configs="$new_configs $c"
	idx=$(($idx + 1))

    c=config.ref.$idx
    echo "OPT_LOOP_PEELING = true" > $c
    new_configs="$new_configs $c"
	idx=$(($idx + 1))

	c=config.ref.$idx
    echo "OPT_REGISTER_BLOCKING = true" > $c
	new_configs="$new_configs $c"
	idx=$(($idx + 1))

	c=config.ref.$idx
    echo "OPT_OFFSET_CSE = true" > $c
	new_configs="$new_configs $c"
	idx=$(($idx + 1))

	c=config.ref.$idx
    echo "OPT_REGISTER_BLOCKING = true" > $c
    echo "OPT_OFFSET_CSE = true" >> $c	
	new_configs="$new_configs $c"
	idx=$(($idx + 1))

	c=config.ref.$idx
    echo "OPT_OFFSET_SPATIAL_CSE = true" > $c	
	new_configs="$new_configs $c"
	idx=$(($idx + 1))

	c=config.ref.$idx
    echo "OPT_REGISTER_BLOCKING = true" > $c
    echo "OPT_OFFSET_COMP = true" >> $c	
	new_configs="$new_configs $c"
	idx=$(($idx + 1))

	c=config.ref.$idx
    echo "OPT_UNCONDITIONAL_GET = true" > $c
	new_configs="$new_configs $c"
	idx=$(($idx + 1))

	c=config.ref.$idx
    echo "OPT_UNCONDITIONAL_GET = true" > $c
    echo "OPT_OFFSET_COMP = true" >> $c	
	new_configs="$new_configs $c"
	idx=$(($idx + 1))

	c=config.ref.$idx
    echo "OPT_UNCONDITIONAL_GET = true" > $c
    echo "OPT_OFFSET_COMP = true" >> $c
    echo "OPT_LOOP_OPT = true" >> $c		
	new_configs="$new_configs $c"
	idx=$(($idx + 1))
	
    echo $new_configs
}

function generate_translation_configurations_cuda()
{
    local configs=""    
    if [ $# -gt 0 ]; then
		configs=$*
    else
		configs=$(generate_empty_translation_configuration)
    fi
    local pre_calc='false'
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
    for config in $new_configs; do
		# OPT_KERNEL_INLINING
        local c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_KERNEL_INLINING = true" >> $c
        new_configs="$new_configs $c"

		# OPT_LOOP_PEELING
        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_LOOP_PEELING = true" >> $c
        new_configs="$new_configs $c"

		# OPT_REGISTER_BLOCKING
        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_REGISTER_BLOCKING = true" >> $c
        new_configs="$new_configs $c"

		# OPT_UNCONDITIONAL_GET
        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_UNCONDITIONAL_GET = true" >> $c
        new_configs="$new_configs $c"

		# OPT_REGISTER_BLOCKING with UNCONDITIONAL_GET
        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_REGISTER_BLOCKING = true" >> $c
        echo "OPT_UNCONDITIONAL_GET = true" >> $c
        new_configs="$new_configs $c"

		# OPT_OFFSET_CSE
        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_OFFSET_CSE = true" >> $c
        new_configs="$new_configs $c"

		# OPT_OFFSET_CSE and OPT_REGISTER_BLOCKING
        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_REGISTER_BLOCKING = true" >> $c
        echo "OPT_OFFSET_CSE = true" >> $c
        new_configs="$new_configs $c"

		# OPT_OFFSET_SPATIAL_CSE
        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_OFFSET_SPATIAL_CSE = true" >> $c
        new_configs="$new_configs $c"

		# OPT_REGISTER_BLOCKING and OPT_OFFSET_COMP
        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_REGISTER_BLOCKING = true" >> $c		
        echo "OPT_OFFSET_COMP = true" >> $c
        new_configs="$new_configs $c"

        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_REGISTER_BLOCKING = true" >> $c		
        echo "OPT_OFFSET_COMP = true" >> $c
        echo "OPT_UNCONDITIONAL_GET = true" >> $c		
        new_configs="$new_configs $c"

        c=config.cuda.$idx
		idx=$(($idx + 1))
        cat $config > $c
        echo "OPT_REGISTER_BLOCKING = true" >> $c		
        echo "OPT_OFFSET_COMP = true" >> $c
        echo "OPT_UNCONDITIONAL_GET = true" >> $c
        echo "OPT_LOOP_OPT = true" >> $c
        new_configs="$new_configs $c"
		
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
		configs=$(generate_empty_translation_configuration)
    fi
	local bsize="64,4,1 32,8,1"
    local overlap='false true'
    local multistream='false true'
    local new_configs=""
    local idx=0
	for l in $bsize; do
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
					echo "CUDA_BLOCK_SIZE = {$l}" >> $c
					echo "MPI_OVERLAP = $i" >> $c
					echo "MULTISTREAM_BOUNDARY = $j" >> $c
					new_configs="$new_configs $c"
				done
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
    LDFLAGS="-L${CMAKE_BINARY_DIR}/runtime -lm"
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
			nvcc -m64 -c $src_file -I${CMAKE_SOURCE_DIR}/include $NVCC_CFLAGS -Xcompiler $(echo $CFLAGS|sed 's/ /,/g') &&
			nvcc -m64 "$src_file_base".o  -lphysis_rt_cuda $LDFLAGS \
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

    #sync
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
    local ref_exe=${CMAKE_CURRENT_BINARY_DIR}/test_cases/$src_name.manual.$target.exe
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
		if ls ${CMAKE_CURRENT_BINARY_DIR}/test_cases/$src_name.manual.*.exe > \
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
	# make sure the binary is availale on each node; without this mpirun often fails
	# if mpi-cuda is used
    $MPIRUN -np $np $mfile_option --output-filename executable-copy $1
    echo "[EXECUTE] $MPIRUN -np $np $mfile_option $* --physis-proc $proc_dim --physis-nlp $PHYSIS_NLP" >&2
    $MPIRUN -np $np $mfile_option $* --physis-proc $proc_dim --physis-nlp $PHYSIS_NLP
}

function execute()
{
    local target=$2
    local exename=$(basename $1 .c).$target.exe
    echo  "[EXECUTE] Executing $exename"
    case $target in
		ref)
			./$exename > $exename.out 2> $exename.err
			;;
		cuda)
			./$exename > $exename.out 2> $exename.err    
			;;
		mpi)
			do_mpirun $3 $4  "$MPI_MACHINEFILE" ./$exename > $exename.out 2> $exename.err    
			;;
		mpi-cuda)
			do_mpirun $3 $4 "$MPI_MACHINEFILE" ./$exename > $exename.out 2> $exename.err    
			;;
		*)
			exit_error "Unsupported target: $2"
			;;
    esac
    if [ $? -ne 0 ] || grep -q "orted was unable to" $exename.err; then
		cat $exename.err
		return 1
    fi
	rm $exename		
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
	rm $exename.out $exename.err	
}

function use_valgrind()
{
    case $1 in
		translate)
			PHYSISC="valgrind --error-exitcode=1 --suppressions=${CMAKE_SOURCE_DIR}/misc/valgrind-suppressions.supp $PHYSISC"
			;;
		execute)
			EXECUTE_WITH_VALGRIND=1
			;;
		*)
			exit_error "Unknown step $1"
			;;
    esac
}

function print_usage()
{
    echo "USAGE"
    echo -e "\trun_system_tests.sh [options]"
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
    echo -e "\t--physis-nlp <number-of-gpus-per-node>"
    echo -e "\t\tNumber of GPUs per node."
    echo -e "\t--machinefile <file-path>"
    echo -e "\t\tThe MPI machinefile."
    echo -e "\t--with-valgrind <translate|execute>"
    echo -e "\t\tExecute the given step with Valgrind. Excution with Valgrind\n\t\tis not yet supported."
    echo -e "\t--priority <level>"
    echo -e "\t\tExecute the test cases with priority higher than or equal\n\t\tto the given level."
	echo -e "\t--config <config-file-path>"
	echo -e "\t\tRead configuration option from the file."
	echo -e "\t-q, --quit"
	echo -e "\t\tQuit immediately upon error."
	echo -e "\t--clear"
	echo -e "\t\tClear output files."
}

function filter_test_case_by_priority()
{
	local priority=$1
	shift
	local tests=$*
	local tests_out=""
	for test in $tests; do
		local test_priority=$(grep PRIORITY $test | awk '{print $3}')
		if [ "x$test_priority" = "x" ]; then continue; fi
		if [ $test_priority -le $priority ]; then
			tests_out+="$test "
		fi
	done
	echo $tests_out
}

function get_test_cases()
{
	local tests=""
    if [ $# -eq 0 ]; then
		tests=$(find ${CMAKE_CURRENT_SOURCE_DIR}/test_cases -name "test_*.c"|sort -n)
    else
		for t in $*; do
			tests+="${CMAKE_CURRENT_SOURCE_DIR}/test_cases/$t.c "
		done
	fi
    set +e	
    tests=$(for t in $tests; do echo -e "$t\n" | grep -v 'test_.*\.manual\.'; done)
    set -e
	echo $tests
}

{
    TARGETS=""
    STAGE="ALL"

    TESTS=$(get_test_cases)
	
    TEMP=$(getopt -o ht:s:m:q --long help,clear,targets:,source:,translate,compile,execute,mpirun,machinefile:,proc-dim:,physis-nlp:,quit,with-valgrind:,priority:,config: -- "$@")
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
				echo $TESTS
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
			--physis-nlp)
				PHYSIS_NLP=$2
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
			--with-valgrind)
				use_valgrind $2
				shift 2
				;;
			--priority)
				PRIORITY=$2
				shift 2
				;;
			--config)
				CONFIG_ARG=$(abs_path $2)
				shift 2
				;;
			-h|--help)
				print_usage
				exit 0
				shift
				;;
			--clear)
				clear_output
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

	echo priority: $PRIORITY
	TESTS=$(filter_test_case_by_priority $PRIORITY $TESTS)
    
	# Test all targets by default
    if [ "x$TARGETS" = "x" ]; then TARGETS=$($PHYSISC --list-targets); fi
    
    echo "Test sources: $(for i in $TESTS; do basename $i; done | xargs)"
    echo "Targets: $TARGETS"

    for TARGET in $TARGETS; do
		for TEST in $TESTS; do
			SHORTNAME=$(basename $TEST)
			DESC=$(grep -o '\WTEST: .*$' $TEST | sed 's/\WTEST: \(.*\)$/\1/')
			DIM=$(grep -o '\WDIM: .*$' $TEST | sed 's/\WDIM: \(.*\)$/\1/')
			if [ "x$CONFIG_ARG" = "x" ]; then
				CONFIG=$(generate_translation_configurations $TARGET)
			else
				CONFIG=$CONFIG_ARG
			fi
			for cfg in $CONFIG; do
				echo "Testing with $SHORTNAME ($DESC)"
				echo "Configuration ($cfg):"
				echo "Dimension: $DIM"
				cat $cfg
				echo "[TRANSLATE] Processing $SHORTNAME for $TARGET target"
				echo $PHYSISC --$TARGET -I${CMAKE_SOURCE_DIR}/include \
					--config $cfg $TEST
				if $PHYSISC --$TARGET -I${CMAKE_SOURCE_DIR}/include \
					--config $cfg $TEST > $(basename $TEST).$TARGET.log \
					2>&1; then
					echo "[TRANSLATE] SUCCESS"
					NUM_SUCCESS_TRANS=$(($NUM_SUCCESS_TRANS + 1))
				else
					echo "[TRANSLATE] FAIL"
					NUM_FAIL_TRANS=$(($NUM_FAIL_TRANS + 1))
					fail $SHORTNAME $TARGET translate $cfg
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
					fail $SHORTNAME $TARGET compile $cfg
					continue				
				fi
				if [ "$STAGE" = "COMPILE" ]; then continue; fi
				np_target=1
				if [ "$TARGET" = "mpi" -o "$TARGET" = "mpi-cuda" ]; then
					np_target=$MPI_PROC_DIM
					echo "[EXECUTE] Trying with process configurations: $np_target"
				fi
				for np in $np_target; do
					if execute $SHORTNAME $TARGET $np $DIM; then
						echo "[EXECUTE] SUCCESS"
						NUM_SUCCESS_EXECUTE=$(($NUM_SUCCESS_EXECUTE + 1))
					else
						echo "[EXECUTE] FAIL"
						NUM_FAIL_EXECUTE=$(($NUM_FAIL_EXECUTE + 1))
						fail $SHORTNAME $TARGET execute $cfg $np
						continue
					fi
				done
				print_results_short			
			done
		done
    done
    finish
} 2>&1 | tee $LOGFILE

# Local Variables:
# mode: sh-mode
# End:


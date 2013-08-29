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
DEFAULT_TARGETS="ref cuda mpi mpi-cuda"
FLAG_KEEP_OUTPUT=1
if [ "x$CFLAGS" = "x" ]; then
    CFLAGS=""
fi
CFLAGS="${CFLAGS} -g -Wno-unused-variable"
DIE_IMMEDIATELY=0
###############################################################
set -u
#set -e
function realpath() {
	local p=$1
	local dname=""
	if echo $p | grep '^/' > /dev/null 2>&1; then
		echo $p
	else
		echo $(pwd)/$p
	fi
}
SELF_NAME=$(realpath $0)
SELF_BASE=$(basename $0)
TIMESTAMP=$(date +%m-%d-%Y_%H-%M-%S)
LOGFILE=$PWD/${SELF_BASE%.sh}.log.$TIMESTAMP
WD=$WD_BASE/$TIMESTAMP
ORIGINAL_DIRECTORY=$PWD

NUM_ALL_TESTS=0
NUM_SUCCESS_TRANS=0
NUM_FAIL_TRANS=0
NUM_SUCCESS_COMPILE=0
NUM_FAIL_COMPILE=0
NUM_SUCCESS_EXECUTE=0
NUM_FAIL_EXECUTE=0
NUM_WARNING=0
FAILED_TESTS=""

PHYSISC=@CMAKE_BINARY_DIR@/translator/physisc
MPIRUN=mpirun
MPI_PROC_DIM="1,1x1,1x1x1" # use only 1 process by default; can be controlled by the -np option
MPI_MACHINEFILE=""
PHYSIS_NLP=1
TRACE=""

EXECUTE_WITH_VALGRIND=0

PRIORITY=1
CONFIG_ARG=""

EMAIL_TO=""

RUN_PARALLEL=0
PARALLEL_NP=""

RETURN_SUCCESS=0
FAIL_TRANSLATE=1
FAIL_COMPILE=2
FAIL_EXECUTE=3
###############################################################

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
    echo  -n "[TRANSLATE] $NUM_SUCCESS_TRANS/$NUM_FAIL_TRANS/$NUM_ALL_TESTS"
    echo  -n ", [COMPILE] $NUM_SUCCESS_COMPILE/$NUM_FAIL_COMPILE/$NUM_ALL_TESTS"
    echo  ", [EXECUTE] $NUM_SUCCESS_EXECUTE/$NUM_FAIL_EXECUTE/$NUM_ALL_TESTS."
}

function email_results()
{
	local subject="Tests report: "
	if [ "x" = "x$FAILED_TESTS" ]; then
		subject+="Successfully completed"
	else
		subject+="$(echo "$FAILED_TESTS" | wc -w) failure(s)!!!"
	fi
	mail -r 'Physis testing <nmaruyama@riken.jp>' -s "$subject" $EMAIL_TO <<EOF
At $TIMESTAMP on `hostname`:

$1
EOF
}

function print_results()
{
	local msg=""
	msg+="[TRANSLATE] $NUM_SUCCESS_TRANS/$NUM_FAIL_TRANS/$NUM_ALL_TESTS."
	msg+="\n[COMPILE]   $NUM_SUCCESS_COMPILE/$NUM_FAIL_COMPILE/$NUM_ALL_TESTS."
	msg+="\n[EXECUTE]   $NUM_SUCCESS_EXECUTE/$NUM_FAIL_EXECUTE/$NUM_ALL_TESTS."
	msg+="\n"
	if [ "x" != "x$FAILED_TESTS" ]; then
		msg+="\n!!! $(echo "$FAILED_TESTS" | wc -w) failure(s)!!!"
		msg+="\nFailed tests: $FAILED_TESTS"
	else
		msg+="\nAll tests completed successfully."
	fi
	if [ $NUM_WARNING -gt 0 ]; then
		msg+="$NUM_WARNING warning(s)"
	fi
	echo -e "$msg"
	msg=$(echo -e "$msg")
	if [ "$EMAIL_TO" != "" ]; then
		email_results "$msg"
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
	rm -f $ORIGINAL_WD/${0%.sh}.log.*
	rm -rf $WD_BASE
}

function abs_path()
{
    local paths=$1
	local ap=""
	for path in $paths; do 
		if [ $(expr substr $path 1 1) = '/' ]; then
			ap="$ap $path"
		else
			ap="$ap $ORIGINAL_WD/$path"
		fi
    done
	echo $ap
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
    local bsize="64,4,1 32,8,1"
    local new_configs=""
    local idx=0
	for j in $bsize; do
		for k in $configs; do
			local c=config.cuda.$idx
			idx=$(($idx + 1))
			cat $k > $c
			echo "CUDA_BLOCK_SIZE = {$j}" >> $c
			new_configs="$new_configs $c"
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

function generate_translation_configurations_cuda_hm()
{
    if [ $# -gt 0 ]; then
		echo $1
    else
		generate_empty_translation_configuration
    fi
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

function generate_translation_configurations_opencl()
{
    local configs=""    
    if [ $# -gt 0 ]; then
	configs=$*
    else
	configs=config.empty
	echo -n "" > config.empty
    fi
    local bsize="64,4,1 32,8,1"
    local new_configs=""
    local idx=0
	for j in $bsize; do
	    for k in $configs; do
		local c=config.opencl.$idx
		idx=$(($idx + 1))
		cat $k > $c
		echo "OPENCL_BLOCK_SIZE = {$j}" >> $c
		new_configs="$new_configs $c"
	    done
	done
    echo $new_configs
}

function generate_translation_configurations_mpi_opencl()
{
    local configs=""    
    if [ $# -gt 0 ]; then
	configs=$*
    else
	configs=config.empty
	echo -n "" > config.empty
    fi
    configs=$(generate_translation_configurations_opencl "$configs")
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
		local c=config.mpi-opencl.$idx
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


function generate_translation_configurations_mpi_openmp()
{
    local configs=""    
    if [ $# -gt 0 ]; then
        configs=$*
    else
        configs=config.empty
        echo -n "" > config.empty
    fi
    local dsize="1,1,1 2,2,2"
    local csize="100,100,100 5,5,5"
    local new_configs=""
    local idx=0
    for j in $dsize; do
      for jj in $csize ; do
        for k in $configs; do
            local c=config.mpi-openmp.$idx
            idx=$(($idx + 1))
            cat $k > $c
            echo "MPI_OPENMP_DIVISION = {$j}" >> $c
            echo "MPI_OPENMP_CACHESIZE = {$jj}" >> $c
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
		cuda-hm)
			generate_translation_configurations_cuda_hm
			;;
		mpi|mpi2)
			generate_translation_configurations_mpi
			;;
		mpi-cuda)
			generate_translation_configurations_mpi_cuda
			;;
		opencl)
			generate_translation_configurations_opencl
			;;
		mpi-opencl)
			generate_translation_configurations_mpi_opencl
			;;
		mpi-openmp | mpi-openmp-numa )
			generate_translation_configurations_mpi_openmp
			;;
		*)
			generate_empty_compile_configuration
			;;
    esac
}

function get_exe_name()
{
	local src=$1
	local target=$2
	local src_file_base=${src%.c*}.$target
	if [ "$target" = "opencl" -o "$target" = "mpi-opencl" ]; then
		echo $src_file_base
	else
		echo "$src_file_base".exe
	fi
}

function get_suffix()
{
	local x=$1
	echo ${x##*.}
}

function c_compile()
{
	local src=$1
	case $src in
		*.c)
			cc $*
			;;
		*.cc)
			c++ $*
			;;
		*)
			print_error "Unknown source language."
			return 1
			;;
	esac
}

function compile()
{
    local src=$1
    local target=$2
	local input_suffix=$(get_suffix $src)
    local src_file_base=${src%.c*}.$target
    if [ "@MPI_FOUND@" = "TRUE" ]; then
		MPI_CFLAGS="-pthread"
		OPENMP_CFLAGS="-fopenmp"
		for mpiinc in $(echo "@MPI_INCLUDE_PATH@" | sed 's/;/ /g'); do
			MPI_CFLAGS+=" -I$mpiinc"
		done
    fi
    local LDFLAGS="-L@CMAKE_BINARY_DIR@/runtime -lm"
    local NVCC_CFLAGS="-arch sm_20"
    local CUDA_LDFLAGS="-lcudart -L@CUDA_RT_DIR@"
	local mod_base_file=$(get_module_base $src)
	local mod_base_obj=$(basename ${mod_base_file%.c}.o)
	if is_module_test $src; then
		if [ ! -f $mod_base_obj ]; then
			echo "[COMPILE] Compiling module base file $(basename $mod_base_file)"
			if ! cc -c $mod_base_file $CFLAGS -o $mod_base_obj; then
				print_error "module base file compilation error"
				return 1
			fi
		else
			echo "[COMPILE] Reusing module base object $mod_base_obj"
		fi
	fi
	local exe_name=$(get_exe_name $src $target)
    case $target in
		ref)
			local src_file=$src_file_base.$input_suffix
			if ! c_compile $src_file -c -I@CMAKE_SOURCE_DIR@/include $CFLAGS; then
				print_error "Physis code comilation failed"
				return 1
			fi
			if is_module_test $src; then
				c++ "$src_file_base".o $mod_base_obj -lphysis_rt_ref $LDFLAGS -o "$src_file_base".exe
			else
				c++ "$src_file_base".o -lphysis_rt_ref $LDFLAGS -o $exe_name
			fi
			if [ $? -ne 0 ]; then
				print_error "Linking failed"
				return 1
			fi
			;;
		cuda|cuda-hm)
			if [ "@CUDA_FOUND@" != "TRUE" ]; then
				echo "[COMPILE] Skipping CUDA compilation (not supported)"
				return 0
			fi
			local src_file="$src_file_base".cu
			if ! nvcc -m64 -c $src_file -I@CMAKE_SOURCE_DIR@/include $NVCC_CFLAGS \
				-Xcompiler $(echo $CFLAGS|sed 's/ /,/g') ; then
				print_error "Physis code comilation failed"
				return 1
			fi
			if is_module_test $src; then
				nvcc -m64 "$src_file_base".o $mod_base_obj -lphysis_rt_cuda \
					$LDFLAGS -o $exe_name
			else 
				nvcc -m64 "$src_file_base".o  -lphysis_rt_cuda $LDFLAGS \
					-o $exe_name
			fi
			if [ $? -ne 0 ]; then
				print_error "Linking failed"
				return 1
			fi
			;;
		mpi|mpi2)
			if [ "@MPI_FOUND@" != "TRUE" ]; then
				echo "[COMPILE] Skipping MPI compilation (not supported)"
				return 0
			fi
			local src_file="$src_file_base".$input_suffix
			c_compile $src_file -c -I@CMAKE_SOURCE_DIR@/include $MPI_CFLAGS $CFLAGS &&
			local lib_name=physis_rt_mpi
			if [ $target = "mpi2" ]; then lib_name=physis_rt_mpi2; fi
			mpic++ "$src_file_base".o -l$lib_name $LDFLAGS -o $exe_name
			;;
		mpi-cuda)
			if [ "@MPI_FOUND@" != "TRUE" -o "@CUDA_FOUND@" != "TRUE" ]; then
				echo "[COMPILE] Skipping MPI-CUDA compilation (not supported)"
				return 0
			fi
			local src_file="$src_file_base".cu
			local MPI_CUDA_CFLAGS=""
			for f in $CFLAGS $MPI_CFLAGS; do
				if echo $f | grep '^-I' > /dev/null; then
					MPI_CUDA_CFLAGS+="$f "
				else
					MPI_CUDA_CFLAGS+="-Xcompiler $f "
				fi
			done
			nvcc -m64 -c $src_file -I@CMAKE_SOURCE_DIR@/include \
				$NVCC_CFLAGS $MPI_CUDA_CFLAGS &&
			mpic++ "$src_file_base".o -lphysis_rt_mpi_cuda $LDFLAGS $CUDA_LDFLAGS \
				-o $exe_name
			;;
		opencl)
			OPENCL_LDFLAGS=-lOpenCL
			if [ "@OPENCL_FOUND@" != "TRUE" ]; then
				echo "[COMPILE] Skipping OpenCL compilation (not supported)"
				return 0
			fi
			src_file="$src_file_base".$input_suffix
			c_compile $src_file -c -I@CMAKE_SOURCE_DIR@/include -I@OPENCL_INCLUDE_PATH@ $CFLAGS &&
			c++ "$src_file_base".o -lphysis_rt_opencl $LDFLAGS $OPENCL_LDFLAGS -o $exe_name
			;;
		mpi-opencl)
			OPENCL_LDFLAGS=-lOpenCL
			if [ "@MPI_FOUND@" != "TRUE" -o "@OPENCL_FOUND@" != "TRUE" ]; then
				echo "[COMPILE] Skipping MPI-OpenCL compilation (not supported)"
				return 0
			fi
			src_file="$src_file_base".$input_suffix
			c_compile $src_file -c -I@CMAKE_SOURCE_DIR@/include -I@OPENCL_INCLUDE_PATH@ $MPI_CFLAGS $CFLAGS &&
			mpic++ "$src_file_base".o -lphysis_rt_mpi_opencl $LDFLAGS $OPENCL_LDFLAGS -o $exe_name
			;;
		mpi-openmp | mpi-openmp-numa )
			if [ "@MPI_FOUND@" != "TRUE" ]; then
				echo "[COMPILE] Skipping MPI-OPENMP compilation (not supported)"
				return 0
			fi
			src_file="$src_file_base".$input_suffix
			LIBRARY=physis_rt_mpi_openmp
			if [ $target = mpi-openmp-numa ] ; then
				if [ "@NUMA_ENABLED@" != "TRUE" ]; then
					echo "[COMPILE] Skipping MPI-OPENMP-NUMA compilation (not supported)"
					return 0
				fi
				LIBRARY=physis_rt_mpi_openmp_numa
				LDFLAGS+=" -lnuma"
			fi	
			c_compile $src_file -c -I@CMAKE_SOURCE_DIR@/include $MPI_CFLAGS $OPENMP_CFLAGS $CFLAGS &&
			mpic++ $OPENMP_CFLAGS "$src_file_base".o -l$LIBRARY $LDFLAGS -o $exe_name
			;;
		
		*)
			exit_error "Unsupported target"
			;;
    esac
}

function get_reference_exe_name()
{
    local src_name=${1%%.*}
    local target=$2
    case $target in
		mpi|mpi2)
			target=ref
			;;
		cuda-hm|mpi-cuda)
			target=cuda
			;;
		opencl)
			target=ref
			;;
		mpi-opencl)
			target=ref
			;;
		mpi-openmp | mpi-openmp-numa )
			target=ref
			;;
    esac
    local ref_exe=@CMAKE_CURRENT_BINARY_DIR@/test_cases/$src_name.manual.$target.exe
    echo $ref_exe
}

function execute_reference()
{
    local src_name=${1%%.*}
    local target=$2
    local ref_exe=$(get_reference_exe_name $1 $2)
	echo $ref_exe
    local ref_out=$ref_exe.out    
    # Do nothing if no reference implementation is found.
    if [ ! -x $ref_exe ]; then
		echo "[EXECUTE] No reference implementation found." >&2
	# Check if other implementation variants exist. If true,
	# warn about the lack of an implementation for this target
		if ls @CMAKE_CURRENT_BINARY_DIR@/test_cases/$src_name.manual.*.exe > \
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
    #    $MPIRUN -np $np $mfile_option --output-filename executable-copy cat $1 > /dev/null 2> /dev/null
    $MPIRUN -np $np $mfile_option cat $1 > /dev/null 2> /dev/null
    echo "[EXECUTE] $MPIRUN -np $np $mfile_option $* --physis-proc $proc_dim --physis-nlp $PHYSIS_NLP" >&2
    $MPIRUN -np $np $mfile_option $* --physis-proc $proc_dim --physis-nlp $PHYSIS_NLP
}

function execute()
{
    local target=$2
    local exename=$(get_exe_name $1 $target)
    echo  "[EXECUTE] Executing $exename"
    case $target in
		ref)
			./$exename $TRACE > $exename.out 2> $exename.err
			;;
		cuda|cuda-hm)
			./$exename $TRACE > $exename.out 2> $exename.err    
			;;
		mpi|mpi2)
			do_mpirun $3 $4 "$MPI_MACHINEFILE" ./$exename $TRACE > $exename.out 2> $exename.err    
			;;
		mpi-cuda)
			do_mpirun $3 $4 "$MPI_MACHINEFILE" ./$exename $TRACE > $exename.out 2> $exename.err    
			;;
		opencl)
			./$exename $TRACE > $exename.out 2> $exename.err
			;;
		mpi-opencl)
			do_mpirun $3 $4 "$MPI_MACHINEFILE" ./$exename $TRACE > $exename.out 2> $exename.err
			;;
		mpi-openmp | mpi-openmp-numa )
			do_mpirun $3 $4 "$MPI_MACHINEFILE" ./$exename $TRACE > $exename.out 2> $exename.err	    
			;;
		*)
			exit_error "Unsupported target: $2"
			;;
    esac
    if [ $? -ne 0 ] || grep -q "orted was unable to" $exename.err; then
		cat $exename.err
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
	rm $exename.out $exename.err	
}

function use_valgrind()
{
    case $1 in
		translate)
			PHYSISC="valgrind --error-exitcode=1 --suppressions=@CMAKE_SOURCE_DIR@/misc/valgrind-suppressions.supp $PHYSISC"
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
	echo -e "\t--list"
    echo -e "\t\tList available test source files."
    echo -e "\t--translate"
    echo -e "\t\tTest only translation and its dependent tasks."
    echo -e "\t--compile"
    echo -e "\t\tTest only compilation and its dependent tasks."
    echo -e "\t--execute"
    echo -e "\t\tTest only execution and its dependent tasks."
    echo -e "\t-m, --mpirun"
    echo -e "\t\tThe mpirun command for testing MPI-based runs."
    echo -e "\t--config <file-path>"
    echo -e "\t\tConfiguration file passed to the translator."
    echo -e "\t--proc-dim <proc-dim-list>"
    echo -e "\t\tProcess dimension. E.g., to run 16 processes, specify this \n\t\toption like '16,4x4,1x4x4'. This way, 16 processes are mapped\n\t\tto the overall problem domain with the decomposition for\n\t\th dimensionality. Multiple values can be passed with quotations."
    echo -e "\t--physis-nlp <number-of-gpus-per-node>"
    echo -e "\t\tNumber of GPUs per node."
    echo -e "\t--machinefile <file-path>"
    echo -e "\t\tThe MPI machinefile."
    echo -e "\t--trace"
	echo -e "\t\tTrace kernel execution"
    echo -e "\t--with-valgrind <translate|execute>"
    echo -e "\t\tExecute the given step with Valgrind. Excution with Valgrind\n\t\tis not yet supported."
    echo -e "\t--priority <level>"
    echo -e "\t\tExecute the test cases with priority higher than or equal\n\t\tto the given level."
	echo -e "\t--config <config-file-path>"
	echo -e "\t\tRead configuration option from the file."
	echo -e "\t--email <email-address>"
	echo -e "\t\tEmail test result"
	echo -e "\t--parallel [max parallel tests]"
	echo -e "\t\tRun tests in parallel"
	echo -e "\t-q, --quit"
	echo -e "\t\tQuit immediately upon error."
	echo -e "\t--clear"
	echo -e "\t\tClear output files."
}

function get_priority()
{
	local test=$1
	local test_priority=$(grep PRIORITY $test | awk '{print $3}')
	echo $test_priority
}

function get_description()
{
	local test=$1
	DESC=$(grep -o '\WTEST: .*$' $test | sed 's/\WTEST: \(.*\)$/\1/')
	echo $DESC
}

function list_tests()
{
	echo "Available test cases"
	for t in $1; do
		local p=$(get_priority $t)
		echo -e "\t$(basename $t) #$p ($(get_description $t))"
	done
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
		tests=$(find @CMAKE_CURRENT_SOURCE_DIR@/test_cases -name "test_*.c*"|sort -n)
    else
		for t in $*; do
			tests+="@CMAKE_CURRENT_SOURCE_DIR@/test_cases/$t "
		done
	fi
    set +e	
    tests=$(for t in $tests; do echo -e "$t\n" | grep -v 'test_.*\.manual\.'; done)
    tests=$(for t in $tests; do echo -e "$t\n" | grep -v 'test_.*\.module_base\.'; done)
	# Filter out emacs backup files
    tests=$(for t in $tests; do echo -e "$t\n" | grep -v '*~'; done)		
    set -e
	echo $tests
}

function get_desc()
{
	local TEST=$1
	local DESC=$(grep -o '\WTEST: .*$' $TEST | sed 's/\WTEST: \(.*\)$/\1/')
	echo $DESC
}

function is_module_test()
{
	echo $1 | grep 'test_.*\.module\.c' > /dev/null 2>&1
}

function is_skipped_module_test()
{
	if is_module_test $1; then
		case "$2" in
			cuda-hm|mpi|mpi2|mpi-cuda|opencl|mpi-opencl|mpi-openmp|mpi-openmp-numa)
				return 0
		esac
	fi
	return 1
}

function get_module_base()
{
	local mod_physis=$1
	echo @CMAKE_CURRENT_SOURCE_DIR@/test_cases/${mod_physis%.module.c}.module_base.c
}

function count_num_all_tests()
{
	local TARGETS=$1
	local TESTS=$2
	local num=0
    for TARGET in $TARGETS; do
		for TEST in $TESTS; do
			SHORTNAME=$(basename $TEST)			
			if is_skipped_module_test $TEST $TARGET; then
				continue;
			fi
			if [ "x$CONFIG_ARG" = "x" ]; then
				CONFIG=$(generate_translation_configurations $TARGET)
			else
				CONFIG=$CONFIG_ARG
			fi
			num=$((num + $(echo $CONFIG | wc -w)))
		done
	done
	echo $num
}

function do_test_finish()
{
    local pass=$1
	echo $pass > result
}

function do_test()
{
	local TARGET=$1
	local TEST=$2
	local DIM=$3
	local SHORTNAME=$4
	local STAGE=$5
	local cfg=$6
	local DESC=$(get_desc $TEST)
	
	local test_wd=$WD/$TARGET/$SHORTNAME/$(basename $cfg)
	mkdir -p $test_wd
	pushd . > /dev/null
	cd $test_wd

	{
		echo "Testing with $SHORTNAME ($DESC)"
		echo "Configuration ($cfg):"
		if [ $cfg = "$(basename $cfg)" ]; then
			cfg=$WD/$cfg
		fi
		cat $cfg	
		echo "Dimension: $DIM"

		echo "[TRANSLATE] Processing $SHORTNAME for $TARGET target"
		#echo $PHYSISC --$TARGET -I@CMAKE_SOURCE_DIR@/include \
		#	--config $cfg $TEST
		if $PHYSISC --$TARGET -I@CMAKE_SOURCE_DIR@/include \
			--config $cfg $TEST > $(basename $TEST).$TARGET.log \
			2>&1; then
			echo "[TRANSLATE] SUCCESS"
		else
			echo "[TRANSLATE] FAIL"
			#fail $SHORTNAME $TARGET translate $cfg
			do_test_finish $FAIL_TRANSLATE
			popd > /dev/null
			return $FAIL_TRANSLATE
		fi
		if [ "$STAGE" = "TRANSLATE" ]; then return; fi
		echo "[COMPILE] Processing $SHORTNAME for $TARGET target"
		if compile $SHORTNAME $TARGET; then
			echo "[COMPILE] SUCCESS"
		else
			echo "[COMPILE] FAIL"
			#fail $SHORTNAME $TARGET compile $cfg
			do_test_finish $FAIL_COMPILE
			popd > /dev/null
			return $FAIL_COMPILE
		fi
		if [ "$STAGE" = "COMPILE" ]; then continue; fi
		
		case "$TARGET" in
			mpi|mpi2|mpi-*)
				np_target=$MPI_PROC_DIM
				;;
			*)
				np_target=1
		esac
		echo "[EXECUTE] Trying with process configurations: $np_target"				
		for np in $np_target; do
			if execute $SHORTNAME $TARGET $np $DIM; then
				execute_success=1
			else
				execute_success=0
			fi
			if [ $execute_success -eq 1 ]; then
				echo "[EXECUTE] SUCCESS"
			else
				echo "[EXECUTE] FAIL"
				#fail $SHORTNAME $TARGET execute $cfg $np
				rm -f $(get_exe_name $SHORTNAME $TARGET)
				do_test_finish $FAIL_EXECUTE
				popd > /dev/null
				return $FAIL_EXECUTE
			fi
		done
		do_test_finish $RETURN_SUCCESS
	} 2>&1 | tee log
	popd > /dev/null
	return 0
}

function inc()
{
	local var=$1
	eval $var=$(($var + 1))
}

function update_results()
{
	local f=$1
	local code=$(cat $f)
	local test_sig=$(dirname $f)
	case $code in
		$RETURN_SUCCESS)
			inc NUM_SUCCESS_TRANS
			inc NUM_SUCCESS_COMPILE
			inc NUM_SUCCESS_EXECUTE
			;;
		$FAIL_TRANSLATE)
			inc NUM_FAIL_TRANS
			;;
		$FAIL_COMPILE)
			inc NUM_FAIL_COMPILE
			;;
		$FAIL_EXECUTE)
			inc NUM_FAIL_EXECUTE
			;;
	esac
	if [ "$code" != $RETURN_SUCCESS ]; then
		FAILED_TESTS="$FAILED_TESTS $test_sig"
	fi
}

function fetch_results()
{
	for f in $(find $1/$2 -name result | sort -n ); do
		update_results $f
		mv $f $f.checked
	done
}

function do_test_parallel()
{
	local TARGET=$1
	local TEST=$2
	local CONFIG=$3
	local DIM=$4
	local SHORTNAME=$5
	local STAGE=$6
	local PARALLEL_OPT=""
	if [ -n "$PARALLEL_NP" ]; then
		local PARALLEL_OPT="-j $PARALLEL_NP"
	fi
	if [ "1" = "$RUN_PARALLEL" ]; then
		if ! type parallel > /dev/null 2>&1; then
			exit_error "Parallel testing requested, but parallel command is not found."
		fi
		parallel $PARALLEL_OPT $SELF_NAME subcommand $WD $LOGFILE do_test $TARGET $TEST $DIM $SHORTNAME $STAGE -- $CONFIG
		fetch_results $TARGET $SHORTNAME
		print_results_short
		if [ $DIE_IMMEDIATELY -eq 1 -a -n "$FAILED_TESTS" ]; then
			exit_error
		fi
	else
		for cfg in $CONFIG; do
			do_test $TARGET $TEST $DIM $SHORTNAME $STAGE $cfg
			fetch_results $TARGET $SHORTNAME
			print_results_short
			if [ $DIE_IMMEDIATELY -eq 1 -a -n "$FAILED_TESTS" ]; then
				exit_error
			fi
		done
	fi
}

function is_sub_command()
{
	if [ "subcommand" = "$1" ]; then
		return 0
	else
		return 1
	fi
}

function do_sub_command()
{
	shift 1
	WD=$1
	shift 1
	LOGFILE=$1
	shift 1
	{
		eval $@
	} 2>&1 | tee -a $LOGFILE
}


if is_sub_command $@; then
	do_sub_command $@
	exit $?
fi

{
	mkdir -p $WD
	ORIGINAL_WD=$(pwd)
	cd $WD
	
    TARGETS=""
    STAGE="ALL"

    TESTS=$(get_test_cases)
	
    TEMP=$(getopt -o ht:s:m:q --long help,clear,targets:,source:,translate,compile,execute,mpirun,machinefile:,proc-dim:,physis-nlp:,quit,with-valgrind:,priority:,trace,config:,email:,list,parallel::, -- "$@")

    if [ $? != 0 ]; then
		print_error "Error in getopt. Invalid options: $@"
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
			--trace)
				TRACE=--physis-trace
				shift
				;;
			--config)
				CONFIG_ARG=$(abs_path "$2")
				shift 2
				;;
			--email)
				EMAIL_TO=$2
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
			--list)
				list_tests "$TESTS"
				exit 0
				;;
			--parallel)
				RUN_PARALLEL=1
				PARALLEL_NP="$2"
				shift 2
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

	TESTS=$(filter_test_case_by_priority $PRIORITY $TESTS)
    
    if [ "x$TARGETS" = "x" ]; then
		TARGETS=$DEFAULT_TARGETS
	fi

	NUM_ALL_TESTS=$(count_num_all_tests "$TARGETS" "$TESTS")
	
    echo "Test sources: $(for i in $TESTS; do basename $i; done | xargs)"
    echo "Targets: $TARGETS"
	echo "Number of tests: $NUM_ALL_TESTS"
	if [ -n "$RUN_PARALLEL" ]; then
		echo -n "Parallel mode"
		if [ -n "$PARALLEL_NP" ]; then
			echo " (up to $PARALLEL_NP parallel tests)"
		else
			echo ""
		fi
	fi

    for TARGET in $TARGETS; do
		for TEST in $TESTS; do
			SHORTNAME=$(basename $TEST)			
			if is_skipped_module_test $TEST $TARGET; then
				echo "Skipping $SHORTNAME for $TARGET target"
				continue;
			fi
			DIM=$(grep -o '\WDIM: .*$' $TEST | sed 's/\WDIM: \(.*\)$/\1/')
			if [ "x$CONFIG_ARG" = "x" ]; then
				CONFIG=$(generate_translation_configurations $TARGET)
			else
				CONFIG=$CONFIG_ARG
			fi
			do_test_parallel $TARGET $TEST "$CONFIG" $DIM $SHORTNAME $STAGE
		done
    done
    finish
} 2>&1 | tee $LOGFILE

# Local Variables:
# mode: sh-mode
# End:

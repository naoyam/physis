#!/usr/bin/env bash
#
# Download, configure, build, and install the latest ROSE library.
#
# Usage:
#
# rose-build.sh [-r <src-path>] [-p <install-path>]
# - src-path: unpacked ROSE source directory
# - install-path: installation path (should be an absolute path)
#
# The JAVA_HOME environment variable must be set to the JDK directory (such as /usr/lib/jvm/java).
#
# After options are determined, each of the following commands can be selected.
# All steps are done by default. 
# - download
# - configure
# - make
# - make install

set -u
set -e

####################################
# Platform specific variables
TOP_DIR=$(pwd)
BOOST=/usr
UNATTENDED=1 # set to 0 to process each step interactively

###################################
while getopts ":r:p:b:" opt; do
    case $opt in
	r)
	    SRC_DIR=$OPTARG
	    ;;
	p)
	    INSTALL_PREFIX=$OPTARG
	    ;;
	b)
	    BOOST=$OPTARG
	    ;;
	\?)
	    die "Invalid option: -$OPTARG"
	    ;;
    esac
done

###################################
TOP_SRC_DIR=$TOP_DIR/src
TOP_BUILD_DIR=$TOP_DIR/build
ROSE_NAME=""
SRC_DIR=""
BUILD_DIR=""
INSTALL_PREFIX=""
#cd $TOP_DIR
mkdir -p $TOP_SRC_DIR
mkdir -p $TOP_BUILD_DIR
####################################
function die()
{
    local msg=$1
    echo "ERROR: $msg"
    exit 1
}

function finish()
{
    echo "Finished successfully."
    exit 0
}
 
function download_latest_tarball()
{
    local site='https://outreach.scidac.gov'
    echo "Detecting the latest ROSE source package..." 
    local path=$(wget --quiet -O- "$site/frs/?group_id=24" | egrep -o -m1 '/frs/download.php/[0-9]+/rose-0.9.5a-without-EDG-[0-9]+\.tar\.gz')
    ROSE_NAME=$(basename $path .tar.gz| sed 's/-without-EDG//')
    local rose_file_name=$(basename $path)
    echo "Latest ROSE: $rose_file_name"
    if [ -f $TOP_DIR/$rose_file_name ]; then
	echo "The latest source is already downloaded"
    else
	echo "Downloading $site$path..."
	wget $site$path
	echo "Download finished."
    fi
    SRC_DIR=$TOP_SRC_DIR/$ROSE_NAME
    if [ -d $SRC_DIR ]; then
	echo "The source is already unpacked."
    else
	echo "Unpacking the source..."
	tar zxf $(basename $path) -C src
    fi
    INSTALL_PREFIX=$TOP_DIR/$ROSE_NAME
    if [ -d $INSTALL_PREFIX ]; then
	echo "Previously installed ROSE of the latest version found."
	echo "Finished successfully."
	exit 0
    fi
    mkdir $INSTALL_PREFIX
    BUILD_DIR=$TOP_BUILD_DIR/$ROSE_NAME
    mkdir $BUILD_DIR
}

function set_boost()
{
    if [ ! -d "$BOOST" ]; then
	die "Boost directory not found ($BOOST)" 
    fi
    if [ ! -d "$BOOST/include" ]; then
	die "Boost header files not found"
    fi
    echo Using Boost at $BOOST
    if [ -e "$BOOST/lib64/libboost_program_options.so" ]; then
        BOOSTLIB=$BOOST/lib64
    elif [ -e "$BOOST/lib/libboost_program_options.so" ]; then
        BOOSTLIB=$BOOST/lib
    else
        die "Boost library not found."
    fi  
}

function set_java_home()
{
    if [ -z "$JAVA_HOME" ]; then
	case $OSTYPE in
	    linux*)
		for i in $(locate libjvm.so); do
		    if echo $i | grep --silent -e gcc -e gcj; then continue; fi
		    echo -n "Using $i? [Y/n] "
		    read yn
		    if [ "$yn" != "n" ]; then
			export JAVA_HOME=${i%/jre*}
			break
		    fi
		done
		
		if [ -z "$JAVA_HOME" ]; then
		    die "JDK not found"
		fi
		;;
	    darwin*)
		JAVA_HOME=/System/Library/Frameworks/JavaVM.framework/Versions/CurrentJDK
		;;
	esac
    fi
    echo JAVA_HOME is set to $JAVA_HOME
}

function detect_java_libraries()
{
    local JVM_DIR=$(dirname $(find ${JAVA_HOME}/ -name "libjvm\.*" | head -1))
    case $OSTYPE in
	linux*)
	    JAVA_LIBRARIES=${JAVA_HOME}/jre/lib:$JVM_DIR
	    ;;
	darwin*)
	    JAVA_LIBRARIES=$JVM_DIR
	    ;;
    esac
}

function exec_configure()
{
    if [ -z "$SRC_DIR" ]; then die "ROSE src path not set"; fi
    if [ -z "$INSTALL_PREFIX" ]; then die "Install prefix not set"; fi
    echo Configuring ROSE at $SRC_DIR
    cd $BUILD_DIR
    echo $SRC_DIR/configure --prefix=$INSTALL_PREFIX --with-CXX_DEBUG=-g --with-CXX_WARNINGS="-Wall -Wno-deprecated" --with-boost=$BOOST --with-boost-libdir=$BOOSTLIB --enable-languages=c,c++,fortran,cuda,opencl --disable-binary-analysis-tests -with-haskell=no 
    if [ $UNATTENDED -ne 1 ]; then
	echo -n "Type Enter to proceed: "
	read x
    fi
    $SRC_DIR/configure --prefix=$INSTALL_PREFIX --with-CXX_DEBUG=-g --with-CXX_WARNINGS="-Wall -Wno-deprecated" --with-boost=$BOOST --with-boost-libdir=$BOOSTLIB --enable-languages=c,c++,fortran,cuda,opencl --enable-binary-analysis-tests=no --disable-projects-directory --disable-tutorial-directory -with-haskell=no 
    if [ $? == 0 ]; then
	echo "Rerun again and select make for building ROSE"
    else
	die "configure failed"
    fi
}

function detect_num_cores()
{
    NUM_PROCESSORS=1 # default
    echo "Detecting number of cores..."
    case $OSTYPE in
	linux*)
	    NUM_PROCESSORS=$(grep processor /proc/cpuinfo|wc -l)
	    ;;
	darwin*)
	    NUM_PROCESSORS=$(system_profiler |grep 'Total Number Of Cores' | awk '{print $5}')
	    ;;
    esac
}

function exec_make()
{
    detect_num_cores
    echo building ROSE by make -j$(($NUM_PROCESSORS / 2))
    cd $BUILD_DIR
    make -j$(($NUM_PROCESSORS / 2))
    if [ $? == 0 ]; then
	echo "Rerun again and select make install to install ROSE"
    fi
}

function exec_install()
{
    echo installing ROSE
    cd $BUILD_DIR
    make install
    echo "ROSE is installed at $INSTALL_PREFIX"
    rm -f $TOP_DIR/latest
    ln -s $INSTALL_PREFIX $TOP_DIR/latest
}

function clean_up_old_files()
{
    cd $TOP_SRC_DIR
    for d in $(ls); do
        if [ ! -d $d ]; then continue; fi
        if [ "$d" = "$ROSE_NAME" ]; then continue; fi
        rm -r $d
    done  
    cd $TOP_BUILD_DIR
    for d in $(ls); do
        if [ ! -d $d ]; then continue; fi
        if [ "$d" = "$ROSE_NAME" ]; then continue; fi
        rm -r $d
    done  
}

####################################

{
    set_boost
#set_java_home
    if [ "x" = "x$JAVA_HOME" ]; then
	die "JAVA_HOME not set"
    fi
    detect_java_libraries

	LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-""}
	DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH:-""}

    case $OSTYPE in
	linux*)
	    echo export LD_LIBRARY_PATH=${JAVA_LIBRARIES}:${BOOST}/lib:$LD_LIBRARY_PATH		
	    export LD_LIBRARY_PATH=${JAVA_LIBRARIES}:${BOOST}/lib:$LD_LIBRARY_PATH
	    ;;
	darwin*)
	    echo export DYLD_LIBRARY_PATH=${JAVA_LIBRARIES}:${BOOST}/lib:$DYLD_LIBRARY_PATH		
	    export DYLD_LIBRARY_PATH=${JAVA_LIBRARIES}:${BOOST}/lib:$DYLD_LIBRARY_PATH
	    ;;
    esac

    command=5
    if [ $UNATTENDED -ne 1 ]; then
	echo "Commands"
	echo "1: download"
	echo "2: configure"
	echo "3: make"
	echo "4: make install"
	echo "5: do all"
	echo -n "What to do? [1-5] (default: 5): "
	read command
    fi
    case $command in
	1)
	    download_latest_tarball
	    ;;
	2)
	    exec_configure
	    ;;
	3)
	    exec_make
	    ;;
	4)
	    exec_install
	    ;;
	5)
	    download_latest_tarball
	    exec_configure
	    exec_make
	    exec_install
            clean_up_old_files
	    ;;
	*)
	    echo Invalid input \"$command\"
	    ;;
    esac

    finish
} 2>&1 | tee rose-build.$(date +%m-%d-%Y_%H-%M-%S).log

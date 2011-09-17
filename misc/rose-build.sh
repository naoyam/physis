#!/usr/bin/env bash
#
# Detects options and path settings and runs configure and make.
# 
# Usage:
# rose-build.sh -r <src-path> -p <install-path>:
# - src-path: unpacked ROSE source directory
# - install-path: installation path (should be an absolute path)
#
# After options are determined, each of the following commands can be selected.
# - configure
# - make
# - make install

#set -u
set -e

ROSE_SRC=""
INSTALL_PREFIX=""

function download_latest_tarball()
{
	local site='https://outreach.scidac.gov'
	echo "Trying to detect the latest ROSE source package..." 
	local path=$(wget --quiet -O- "$site/frs/?group_id=24" | egrep -o -m1 '/frs/download.php/[0-9]+/rose-0.9.5a-without-EDG-[0-9]+\.tar\.gz')
	echo "Downloading $site$path..."
	wget $site$path
	echo "Download finished."
	tar zxf $(basename $path)
	local dname=$(basename $path .tar.gz| sed 's/-without-EDG//')
	mv  $dname src
	mkdir $dname
	mv src $dname
	mkdir $dname/build $dname/install
	ROSE_SRC=$(pwd)/$dname/src
	INSTALL_PREFIX=$(pwd)/$dname/install
	cd $dname/build	
}

function detect_boost()
{
	local BOOST_CANDIDATES=($HOME/homebrew /usr
		$HOME/tools/boost /work0/GSIC/apps/boost/1_45_0/gcc)
	if [ -z "$BOOST" ]; then
		for c in ${BOOST_CANDIDATES[*]}; do
			if [ -d $c/include/boost ]; then
				BOOST=$c
				break
			fi
		done
	fi
	echo Using Boost found at $BOOST
}

function set_java_home()
{
	if [ -z "$JAVA_HOME" ]; then
		case $OSTYPE in
			linux*)
				for i in $(locate libjvm.so); do
					if echo $i | grep --silent -e gcc -e gcj -e openjdk; then continue; fi
					echo -n "Using $i? [Y/n] "
					read yn
					if [ "$yn" != "n" ]; then
						export JAVA_HOME=${i%/jre*}
						break
					fi
				done
				
				if [ -z "$JAVA_HOME" ]; then
					echo "Error: no Java found"
					exit 1
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
	if [ -z "$ROSE_SRC" ]; then echo "ERROR! ROSE src path not set"; exit 1; fi
	if [ -z "$INSTALL_PREFIX" ]; then echo "ERROR! install prefix not set"; exit 1; fi	
	echo Building ROSE at $ROSE_SRC and installing it to $INSTALL_PREFIX
	echo $ROSE_SRC/configure --prefix=$INSTALL_PREFIX --with-CXX_DEBUG=-g --with-CXX_WARNINGS="-Wall -Wno-deprecated" --with-boost=$BOOST --enable-languages=c,c++,fortran,cuda,opencl --disable-binary-analysis-tests -with-haskell=no 
	echo -n "Type Enter to proceed: "
	read x
	$ROSE_SRC/configure --prefix=$INSTALL_PREFIX --with-CXX_DEBUG=-g --with-CXX_WARNINGS="-Wall -Wno-deprecated" --with-boost=$BOOST --enable-languages=c,c++,fortran,cuda,opencl --enable-binary-analysis-tests=no --disable-projects-directory --disable-tutorial-directory -with-haskell=no 
	if [ $? == 0 ]; then
		echo "Rerun again and select make for building ROSE"
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
	echo building ROSE by make -j$NUM_PROCESSORS
	make -j$((NUM_PROCESSORS / 2))
	if [ $? == 0 ]; then
		echo "Rerun again and select make install to install ROSE"
	fi
}

function exec_install()
{
	echo installing ROSE
	make install
	echo "ROSE is installed at $INSTALL_PREFIX"
}

while getopts ":r:p:b:" opt; do
	case $opt in
		r)
			ROSE_SRC=$OPTARG
			;;
		p)
			INSTALL_PREFIX=$OPTARG
			;;
		b)
			BOOST=$OPTARG
			;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			;;
	esac
done

detect_boost
set_java_home
detect_java_libraries

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

echo "Commands"
echo "1: download"
echo "2: configure"
echo "3: make"
echo "4: make install"
echo "5: do all"
echo -n "What to do? [1-5] (default: 5): "
read command
if [ "x$command" = "x" ]; then command="5"; fi
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
		;;
	*)
		echo Invalid input \"$command\"
		;;
esac

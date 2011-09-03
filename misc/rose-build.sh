#!/usr/bin/env bash
#
# Detects options and path settings and runs configure and make.
# 
# Usage:
# rose-build.sh <src-path> <install-path>:
# - src-path: unpacked ROSE source directory
# - install-path: installation path (should be an absolute path)
#
# After options are determined, each of the following commands can be selected.
# - configure
# - make
# - make install

#set -u
set -e

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

shift $(($OPTIND - 1))

if [ -z "$ROSE_SRC" ]; then
	echo -n "ROSE source path: "
	read ROSE_SRC
fi
if [ -z "$INSTALL_PREFIX" ]; then
	echo -n "Install prefix path: "
	read INSTALL_PREFIX
fi

echo Building ROSE at $ROSE_SRC and installing it to $INSTALL_PREFIX
CONFIG_OPTIONS=$*

BOOST_CANDIDATES=($HOME/homebrew /usr $HOME/tools/boost /work0/GSIC/apps/boost/1_45_0/gcc)

if [ -z "$BOOST" ]; then
	for c in ${BOOST_CANDIDATES[*]}; do
		if [ -d $c/include/boost ]; then
			BOOST=$c
			break
		fi
	done
fi

echo Using Boost found at $BOOST

if [ -z "$JAVA_HOME" ]; then
	case $OSTYPE in
		linux*)
			LDFLAGS=""
			for i in $(locate libjvm.so); do
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

JVM_DIR=$(dirname $(find ${JAVA_HOME}/ -name "libjvm\.*" | head -1))
case $OSTYPE in
	linux*)
		JAVA_LIBRARIES=${JAVA_HOME}/jre/lib:$JVM_DIR
		;;
	darwin*)
		JAVA_LIBRARIES=$JVM_DIR
		;;
esac

# Is this necessary?
#export LDFLAGS="-Wl,-rpath=${JAVA_LIBRARIES}:${BOOST}/lib"
case $OSTYPE in
	linux*)
		export LD_LIBRARY_PATH=${JAVA_LIBRARIES}:${BOOST}/lib:$LD_LIBRARY_PATH
		;;
	darwin*)
		export DYLD_LIBRARY_PATH=${JAVA_LIBRARIES}:${BOOST}/lib:$DYLD_LIBRARY_PATH
		;;
esac

echo LDFLAGS is set to $LDFLAGS
echo LD_LIBRARY_PATH is set to $LD_LIBRARY_PATH
echo JAVA_HOME is set to $JAVA_HOME
echo set LD_LIBRARY_PATH to $LD_LIBRARY_PATH

case $OSTYPE in
	linux*)
		NUM_PROCESSORS=$(grep processor /proc/cpuinfo|wc -l)
		;;
	darwin*)
		NUM_PROCESSORS=$(system_profiler |grep 'Total Number of Cores')		
		;;
esac

echo $NUM_PROCESSORS cores detected

echo "Commands"
echo "1: configure"
echo "2: make"
echo "3: make install"
echo -n "What to do? [1-3] "
read command
case $command in
	1)
		echo Configuring ROSE: $ROSE_SRC/configure --prefix=$INSTALL_PREFIX --with-CXX_DEBUG=-g --with-CXX_WARNINGS="-Wall -Wno-deprecated" --with-boost=$BOOST --enable-languages=c,c++,fortran,binaries --enable-cuda --enable-opencl --disable-binary-analysis-tests  $CONFIG_OPTIONS
		echo -n "Type Enter to proceed: "
		read x
		$ROSE_SRC/configure --prefix=$INSTALL_PREFIX --with-CXX_DEBUG=-g --with-CXX_WARNINGS="-Wall -Wno-deprecated" --with-boost=$BOOST --enable-languages=c,c++,fortran,binaries --enable-cuda --enable-opencl --disable-binary-analysis-tests $CONFIG_OPTIONS
		if [ $? == 0 ]; then
			echo "Rerun again and select make for building ROSE"
		fi
		;;
	2)
		echo building ROSE by make -j$((NUM_PROCESSORS / 2))
		make -j$((NUM_PROCESSORS / 2))
		if [ $? == 0 ]; then
			echo "Rerun again and select make install to install ROSE"
		fi
		;;
	3)
		echo installing ROSE
		make install
		;;
	*)
		echo Invalid input \"$command\"
		;;
esac

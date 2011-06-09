#!/usr/bin/env bash

#set -u
set -e

if [ $# -gt 0 ]; then
	ROSE_SRC=$1
	shift
else
	echo -n "ROSE source path: "
	read ROSE_SRC
fi
if [ $# -gt 0 ]; then
	INSTALL_PREFIX=$1
	shift
else
	echo -n "Install prefix path: "
	read INSTALL_PREFIX
fi
echo Building ROSE at $ROSE_SRC and installing it to $INSTALL_PREFIX
CONFIG_OPTIONS=$*

BOOST_CANDIDATES=($HOME/homebrew /usr $HOME/tools/boost /work0/GSIC/apps/boost/1_45_0/gcc)

for c in ${BOOST_CANDIDATES[*]}; do
	if [ -d $c/include/boost ]; then
		BOOST=$c
		break
	fi
done

if [ -z "$JAVA_HOME" ]; then
	case $OSTYPE in
		linux*)
			LDFLAGS=""
			for i in $(locate libjvm.so | grep sun); do
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

echo Configuring ROSE: $ROSE_SRC/configure --prefix=$INSTALL_PREFIX --with-CXX_DEBUG=-g --with-CXX_WARNINGS="-Wall -Wno-deprecated" --with-boost=$BOOST --enable-languages=c,c++,fortran,binaries --enable-cuda --enable-opencl --disable-binary-analysis-tests  $CONFIG_OPTIONS
echo -n "Type Enter to proceed: "
read x
$ROSE_SRC/configure --prefix=$INSTALL_PREFIX --with-CXX_DEBUG=-g --with-CXX_WARNINGS="-Wall -Wno-deprecated" --with-boost=$BOOST --enable-languages=c,c++,fortran,binaries --enable-cuda --enable-opencl --disable-binary-analysis-tests $CONFIG_OPTIONS

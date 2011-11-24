# ROSE Installation

## Prerequisites
* Java JDK (not just JRE)
* Boost C++ library
    * Versions from 1.36 to 1.45
* GNU libtool
* Ghostscript (optional)
* Doxygen (optional)

### Platform specific notes
* Ubuntu/Debian packages
    * sun-java6, libboost-all-dev
* Mac
    * Only OSX 10.6 is suppoted (10.7 not supported)
    * Get Java from the Java developer package from http://connect.apple.com

## Automated way

1. Set the JAVA_HOME environment variable. It should be like /usr/lib/jvm/java (RHEL6/CentOS6/SL6), /usr/lib/jvm/java-6-sun, or /usr/lib/jvm/java-6-sun.
2. Run misc/rose-build.sh. It will automatically download, configure, and compile the ROSE source as suitable for Physis. Just run the script, and the ROSE library will be built under a directory named rose-VERSION/build and installed to rose-VERSION/install. 

## Manual way

Clone the Git repository or download a tar package from the ROSE website.

1. (Git only) run build.sh
2. Linux only
* set JAVA_HOME
* set LD_LIBRARY_PATH to include libjvm.so directory
    * e.g., export LD_LIBRARY_PATH=/usr/lib/jvm/java-6-sun/jre/lib/amd64/server:$LD_LIBRARY_PATH
3. Download the source from https://outreach.scidac.gov/frs/?group_id=24
    * Rose is automatically packaged every week. Newer versions seem
    to be more reliable.
4. unpack the source
5. mkdir <some-build-directory>
6. do configure in the build directory
* --prefix=somewhere
* --with-CXX_DEBUG="-g"
* --with-CXX_WARNINGS="-Wall -Wno-deprecated"
* --enable-cuda
* --with-boost=/usr
    * need to be passed; probably a bug
7. make
    * Hint: use -j option to speed up compilation
8. make install

## Errors and workarounds
### Installing librose.a fails
Edit the top-level libtool on the build directory so that the
compiler_lib_search_path variable (located in the very end of
the file) include the installation lib directory.

### OpenMPI parsing
To parse OpenMPI's header files, option --disable-visibility needs to
be passed to the configure script of OpenMPI, or define
OMPI_C_HAVE_VISIBILITY 0 before including mpi.h.

See https://mailman.nersc.gov/pipermail/rose-public/2010-July/000314.html

# Usage
* Declaration modifier
    * Use SgDeclarationStatement::get_declarationModifier() to get
      SgDeclarationModifier 
    * SgDeclarationModifier::get_storageModifier() to
      getSgStorageModifier
    * Use SgStorageModifier::setStatic
* Do not insert statements and expressions while traversing. Already
  traversed nodes seem fine to edit, but editing remaining nodes
  results in backend errors.
  
# Questions
* SageInterface::lookupNamedTypeInParentScopes
    * The doxygen manual says it does bottom up search. What is the top
    of the stack? Is it the most inner scope? Or is it the root of the
    scope? 

# Misc
* frontend
  * Declared in roseSupport/utility_functions.h

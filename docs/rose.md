# ROSE Installation

## Prerequisites
### Ubuntu
* sun-java6 (JDK, not just JRE)
* libboost-all-dev
    * Only version 1.36 to 1.45 are supported.
* doxygen (optional)

### Mac
* Mac OSX 10.6 (10.7 not supported)
* Java developer package from http://connect.apple.com
* Boost
    * Only version 1.36 to 1.45 are supported.
* Ghostscript (optional)
* GNU libtool

## Automated way

The helper script located at misc/rose-build.sh can be used to
download, configure, and compile the ROSE source as suitable for
Physis. Just run the script, and the ROSE library will be build under
a directory at the current working directory. 

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

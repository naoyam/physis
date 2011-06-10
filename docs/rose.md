# Installation

## Prerequisites
### Ubuntu
* sun-java6 (JDK, not just JRE)
* libboost-all-dev
* doxygen (optional)

### Mac
* Java developer package from http://connect.apple.com
* Brew packages: boost, ghostscript
* GNU libtool

## Steps

There are helper scipts at misc/rose-configure.sh and
misc/rose-build.sh. Both of them automate the following steps. 

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

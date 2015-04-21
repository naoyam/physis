# ROSE Installation

Physis uses the ROSE compiler framework. The current version is tested with the EDG4x version of the ROSE. See below for how to set up a ROSE installation.

## Prerequisites
* Java JDK (not just JRE)
    * Version 1.7 or newer
* Boost C++ library
    * Versions from 1.36 to 1.47
* GNU libtool
* Ghostscript (optional)
* Doxygen (optional)

### Platform specific notes
* Ubuntu/Debian packages
    * sun-java7, libboost-all-dev

## Automated way

1. Set the JAVA_HOME environment variable. It should be like /usr/lib/jvm/java (RHEL6/CentOS6/SL6), /usr/lib/jvm/java-7-sun, or /usr/lib/jvm/java-7-sun.
2. Get the install script, rose-install.sh, from https://github.com/naoyam/rose-tools.
3. Run the script.sh as:

        ./rose-install.sh -g edg4x-rose -b BOOST_INSTALL_DIR
    
    where BOOST_INSTALL_DIR points the directory where the Boost library is installed. If omitted, the sytem default is used. The script automatically download, configure, and compile the ROSE source as suitable for Physis. Just run the script, and the ROSE library will be built under a directory named edg4x-rose/latest.

## Manual way

1. Set JAVA_HOME and LD_LIBRARY_PATH appropriately. LD_LIBRARY_PATH should include libjvm.so directory, e.g.:

         export LD_LIBRARY_PATH=/usr/lib/jvm/java-7-sun/jre/lib/amd64/server:$LD_LIBRARY_PATH
    
2. Download the source from the Github (http://github.com/rose-compiler).
    * There are two versions. The repository at github.com/rose-compiler/edg4x-rose uses the latest EDG4 frontend for parsing C/C++, and is considered the current default version. Another repository at github.com/rose-compiler/rose uses the older EDG3, and is not maintainted anymore. Physis is tested with the edg4x-rose repository.
    * Note that the old tgz packages are not distributed anymore.
3. run build.sh on the top-level source directory.
4. mkdir <some-build-directory>
5. do configure in the build directory
    * --prefix=somewhere
    * --with-CXX_DEBUG="-g"
    * --with-CXX_WARNINGS="-Wall -Wno-deprecated"
    * --with-boost=/usr (depends on your boost installation)
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
* Use SageInterface's methods for manipulating AST nodes and
  edges. node->append_statement() seems to be just connecting an edge
  from the parent to the child, but the pointer from the child to the
  parent is not set correctly.
* Subclasses of AstAttribute must override virtual function copy for
  the deepcopy mechanism copies the attributes as well.
  
# Misc
* frontend and backend functions
  * Declared in roseSupport/utility_functions.h

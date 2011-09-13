#ifndef PHYSIS_CONFIG_H_
#define PHYSIS_CONFIG_H_

// These are duplicates of common/config.h, but need to be here since
// common/config.h is not going to be installed, while this header file
// is installed by make install.
#cmakedefine CUDA_ENABLED
#cmakedefine MPI_ENABLED
#cmakedefine PS_DEBUG
#cmakedefine PS_VERBOSE_DEBUG
#cmakedefine PS_VERBOSE

#cmakedefine AUTO_DOUBLE_BUFFERING

#endif /* PHYSIS_CONFIG_H_ */

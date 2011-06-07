#include <stdlib.h>
#include "log4cpp.h"
#include <iostream>
using namespace std;

struct foo
{
	ostream& print(std::ostream& os) const {
        os << "foo";
        return os;
    }
};

std::ostream &operator<<(std::ostream &os, const foo&x) 
{
    return x.print(os);
}
    

int main(int argc, char *argv[])
{
    cout << "Hello, world!" << endl;

    LOG_DEBUG("test");
    LOG_DEBUG(1.2);
    LOG_DEBUG(string("string"));
    foo x;
    LOG_DEBUG(x);

    LOG_ERROR(x);
    LOG_WARNING("abc");
    TRACE_START;
    
    return EXIT_SUCCESS;
}



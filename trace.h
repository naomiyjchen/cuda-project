#ifndef TRACE_H
#define TRACE_H

#include <iostream>

#ifdef DEBUG
    #define TRACE(msg) \
                std::cerr << "[TRACE] " << __FILE__ << ":" << __LINE__ << " (" << __func__ << ") - " << msg << std::endl;
#else
    #define TRACE(msg) // No operation when DEBUG is not defined
#endif


#endif // TRACE_H

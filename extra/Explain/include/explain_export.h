
#ifndef EXPLAIN_EXPORT_H
#define EXPLAIN_EXPORT_H

#ifdef EXPLAIN_STATIC_DEFINE
#  define EXPLAIN_EXPORT
#  define EXPLAIN_NO_EXPORT
#else
#  ifndef EXPLAIN_EXPORT
#    ifdef explain_EXPORTS
        /* We are building this library */
#      define EXPLAIN_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define EXPLAIN_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef EXPLAIN_NO_EXPORT
#    define EXPLAIN_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef EXPLAIN_DEPRECATED
#  define EXPLAIN_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef EXPLAIN_DEPRECATED_EXPORT
#  define EXPLAIN_DEPRECATED_EXPORT EXPLAIN_EXPORT EXPLAIN_DEPRECATED
#endif

#ifndef EXPLAIN_DEPRECATED_NO_EXPORT
#  define EXPLAIN_DEPRECATED_NO_EXPORT EXPLAIN_NO_EXPORT EXPLAIN_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef EXPLAIN_NO_DEPRECATED
#    define EXPLAIN_NO_DEPRECATED
#  endif
#endif

#endif /* EXPLAIN_EXPORT_H */

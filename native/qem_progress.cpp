// Progress adapter for the installed MIT-licensed Fast-Quadric core.
// The build tool inserts observation hooks, without changing collapse decisions.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <chrono>
#include <cstdint>
#include <climits>
#include <mutex>
#include <new>
static PyObject *observer = nullptr;
static std::mutex job_mutex;
static std::chrono::steady_clock::time_point last_report;
static void meshmill_progress(const char *stage, int done, int total) {
    const auto now = std::chrono::steady_clock::now();
    if (!observer || (done != 0 && done != total &&
        now - last_report < std::chrono::milliseconds(100))) return;
    last_report = now;
    const auto gil = PyGILState_Ensure();
    PyObject *result = PyObject_CallFunction(observer, "sii", stage, done, total);
    if (!result) PyErr_WriteUnraisable(observer);
    Py_XDECREF(result);
    PyGILState_Release(gil);
}
#include "Simplify.h"

static PyObject *reduce_mesh(PyObject *, PyObject *args) {
    PyObject *points, *faces, *callback;
    int target, border;
    double aggression;
    if (!PyArg_ParseTuple(args, "OOidiO", &points, &faces, &target,
                          &aggression, &border, &callback)) return nullptr;
    Py_buffer p{}, f{};
    if (PyObject_GetBuffer(points, &p, PyBUF_CONTIG_RO) < 0) return nullptr;
    if (PyObject_GetBuffer(faces, &f, PyBUF_CONTIG_RO) < 0) {
        PyBuffer_Release(&p); return nullptr;
    }
    const Py_ssize_t np = p.len / (3 * sizeof(double));
    const Py_ssize_t nf = f.len / (3 * sizeof(int64_t));
    if (np > INT_MAX || nf > INT_MAX || target < 0 || target > nf) {
        PyBuffer_Release(&p); PyBuffer_Release(&f);
        return PyErr_Format(PyExc_ValueError, "Invalid mesh size or target");
    }
    PyObject *out_p = nullptr, *out_f = nullptr;
    const char *failure = nullptr;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::lock_guard<std::mutex> lock(job_mutex);
        observer = callback == Py_None ? nullptr : callback;
        last_report = {};
        const auto *xyz = static_cast<const double *>(p.buf);
        const auto *idx = static_cast<const int64_t *>(f.buf);
        Simplify::vertices.clear(); Simplify::triangles.clear();
        Simplify::vertices.reserve(np); Simplify::triangles.reserve(nf);
        for (int i = 0; i < np; ++i) {
            Simplify::Vertex v;
            v.p.x = xyz[3*i]; v.p.y = xyz[3*i+1]; v.p.z = xyz[3*i+2];
            Simplify::vertices.push_back(v);
            if ((i & 16383) == 0) meshmill_progress("Preparing QEM vertices", i, np);
        }
        meshmill_progress("Preparing QEM vertices", np, np);
        for (int i = 0; i < nf; ++i) {
            Simplify::Triangle t;
            t.attr = 0; t.material = -1;
            for (int j = 0; j < 3; ++j) t.v[j] = int(idx[3*i+j]);
            Simplify::triangles.push_back(t);
            if ((i & 16383) == 0) meshmill_progress("Preparing QEM triangles", i, nf);
        }
        meshmill_progress("Preparing QEM triangles", nf, nf);
        Simplify::simplify_mesh(target, aggression, false, border != 0);
        const auto gil = PyGILState_Ensure();
        out_p = PyBytes_FromStringAndSize(nullptr, Simplify::vertices.size()*3*sizeof(double));
        out_f = PyBytes_FromStringAndSize(nullptr, Simplify::triangles.size()*3*sizeof(int32_t));
        if (out_p && out_f) {
            auto *pv = reinterpret_cast<double *>(PyBytes_AS_STRING(out_p));
            auto *fv = reinterpret_cast<int32_t *>(PyBytes_AS_STRING(out_f));
            for (size_t i=0; i<Simplify::vertices.size(); ++i) {
                const auto &v = Simplify::vertices[i];
                pv[3*i]=v.p.x; pv[3*i+1]=v.p.y; pv[3*i+2]=v.p.z;
            }
            for (size_t i=0; i<Simplify::triangles.size(); ++i)
                for (int j=0;j<3;++j) fv[3*i+j]=Simplify::triangles[i].v[j];
        }
        observer = nullptr;
        PyGILState_Release(gil);
    } catch (const std::bad_alloc &) {
        observer = nullptr;
        failure = "Insufficient memory for QEM reduction";
    } catch (...) {
        observer = nullptr;
        failure = "Native QEM reduction failed";
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&p); PyBuffer_Release(&f);
    if (failure) {
        Py_XDECREF(out_p); Py_XDECREF(out_f);
        return PyErr_Format(PyExc_RuntimeError, "%s", failure);
    }
    if (!out_p || !out_f) { Py_XDECREF(out_p); Py_XDECREF(out_f); return PyErr_NoMemory(); }
    return Py_BuildValue("NN", out_p, out_f);
}
static PyMethodDef methods[] = {{"simplify", reduce_mesh, METH_VARARGS, "Reduce with native progress."}, {nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_meshmill_qem", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__meshmill_qem() { return PyModule_Create(&module); }

"""Read actual CUDA/EGL device identity without changing device selection."""
import ctypes as C
import json
import os
import sys
import uuid


def cuda_identity():
    lib=C.CDLL('libcuda.so.1')
    assert lib.cuInit(0)==0
    count=C.c_int();assert lib.cuDeviceGetCount(C.byref(count))==0
    rows=[]
    for ordinal in range(count.value):
        device=C.c_int();assert lib.cuDeviceGet(C.byref(device),ordinal)==0
        buf=(C.c_ubyte*16)()
        fn=getattr(lib,'cuDeviceGetUuid_v2',lib.cuDeviceGetUuid)
        assert fn(C.byref(buf),device)==0
        bus=C.create_string_buffer(64);assert lib.cuDeviceGetPCIBusId(bus,64,device)==0
        name=C.create_string_buffer(128);assert lib.cuDeviceGetName(name,128,device)==0
        rows.append(dict(ordinal=ordinal,uuid=str(uuid.UUID(bytes=bytes(buf))),pci=bus.value.decode(),name=name.value.decode()))
    return rows


def egl_function(name,result,*args):
    lib=C.CDLL('libEGL.so.1');lib.eglGetProcAddress.restype=C.c_void_p
    addr=lib.eglGetProcAddress(name.encode())
    if not addr:raise RuntimeError('EGL function unavailable: '+name)
    return C.CFUNCTYPE(result,*args)(addr)


def egl_uuid(device):
    fn=egl_function('eglQueryDeviceBinaryEXT',C.c_uint,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.POINTER(C.c_int))
    value=(C.c_ubyte*16)();size=C.c_int()
    assert fn(device,0x335C,16,C.byref(value),C.byref(size)) and size.value==16
    return str(uuid.UUID(bytes=bytes(value)))


def egl_devices():
    fn=egl_function('eglQueryDevicesEXT',C.c_uint,C.c_int,C.POINTER(C.c_void_p),C.POINTER(C.c_int))
    devices=(C.c_void_p*64)();count=C.c_int();assert fn(64,devices,C.byref(count))
    rows=[]
    for i in range(count.value):
        try:rows.append(dict(egl_index=i,uuid=egl_uuid(devices[i])))
        except Exception as e:rows.append(dict(egl_index=i,error=repr(e)))
    return rows


def current_egl_uuid():
    lib=C.CDLL('libEGL.so.1');lib.eglGetCurrentDisplay.restype=C.c_void_p
    display=lib.eglGetCurrentDisplay();assert display
    fn=egl_function('eglQueryDisplayAttribEXT',C.c_uint,C.c_void_p,C.c_int,C.POINTER(C.c_ssize_t))
    device=C.c_ssize_t();assert fn(display,0x322C,C.byref(device))
    return egl_uuid(device.value)


if __name__=='__main__':
    print(json.dumps(dict(pid=os.getpid(),cuda=cuda_identity(),egl=egl_devices() if '--egl' in sys.argv else None)))

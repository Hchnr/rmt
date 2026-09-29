"""Isolated subprocess using EvalScope's official LCB runner and OS restrictions."""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import resource
import sys


def restrict_process():
    libc=ctypes.CDLL(None,use_errno=True)
    sec=ctypes.CDLL(ctypes.util.find_library('seccomp'))
    vms=int(Path('/proc/self/statm').read_text().split()[0])*os.sysconf('SC_PAGE_SIZE')
    resource.setrlimit(resource.RLIMIT_AS,(vms+2*1024**3,vms+2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    # The evaluation parent may run as root. Generated programs must not retain
    # capabilities that can change another process or bypass their own limits.
    if os.geteuid()==0:
        os.setgroups([])
        os.setresgid(65534,65534,65534)
        os.setresuid(65534,65534,65534)
    class Ruleset(ctypes.Structure): _fields_=[('handled',ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
    rules=Ruleset((1<<13)-1)
    fd=libc.syscall(444,ctypes.byref(rules),ctypes.sizeof(rules),0)
    if fd<0: raise OSError(ctypes.get_errno(),'Landlock unavailable')
    paths=['/usr','/lib','/lib64',str(Path(sys.prefix)/'lib')]
    for path in paths:
        if not Path(path).exists():continue
        parent=os.open(path,os.O_PATH|os.O_CLOEXEC)
        rule=PathRule((1<<2)|(1<<3),parent)
        if libc.syscall(445,fd,1,ctypes.byref(rule),0)<0:raise OSError(ctypes.get_errno(),'Landlock rule')
        os.close(parent)
    if libc.prctl(38,1,0,0,0)!=0:raise OSError('no_new_privs')
    if libc.syscall(446,fd,0)<0:raise OSError(ctypes.get_errno(),'Landlock restrict')
    os.close(fd)
    sec.seccomp_init.argtypes=[ctypes.c_uint32];sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p]
    sec.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=sec.seccomp_init(0x7fff0000)
    for name in ['socket','connect','bind','listen','accept','accept4','sendto','sendmsg','ptrace',
                 'process_vm_readv','process_vm_writev','pidfd_open','pidfd_send_signal','rt_sigqueueinfo','rt_tgsigqueueinfo','mount','umount2','unshare','setns','bpf',
                 'execve','execveat','fork','vfork','clone','clone3','kill','tkill','tgkill',
                 'truncate','ftruncate','open_by_handle_at','io_uring_setup','prlimit64','setrlimit',
                 'chmod','fchmod','fchmodat','fchmodat2','chown','fchown','lchown','fchownat',
                 'utime','utimes','futimesat','utimensat','chroot','capset']:
        nr=sec.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and sec.seccomp_rule_add(ctx,0x50000|1,nr,0)!=0:raise RuntimeError('seccomp rule')
    if sec.seccomp_load(ctx)!=0:raise RuntimeError('seccomp load')
    sec.seccomp_release(ctx)
    os.environ.clear()


def main():
    payload=json.load(sys.stdin)
    from evalscope.benchmarks.live_code_bench.testing_util import run_test
    # Resolve and load seccomp before the filesystem restriction.
    ctypes.CDLL(ctypes.util.find_library('seccomp'))
    restrict_process()
    if payload.get('probe'):
        import socket
        denied=[]
        for label,call in [('file',lambda:open('/etc/passwd').read()),('network',lambda:socket.socket()),
                           ('write',lambda:open('/tmp/rmt_sandbox_should_not_exist','w'))]:
            try: call()
            except (OSError,PermissionError): denied.append(label)
        print(json.dumps({'denied':denied,'euid':os.geteuid(),'egid':os.getegid()}));return
    result,meta=run_test(payload['sample'],test=payload['generation'],timeout=payload.get('timeout',6),debug=False)
    print(json.dumps({'result':[bool(x) if type(x).__name__=='bool' or type(x).__name__=='bool_' else int(x) for x in result],
                      'metadata':meta},default=str))


if __name__=='__main__':main()

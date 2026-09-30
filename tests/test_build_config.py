"""Build configuration checks without compiling CUDA or mutating dependencies."""
import builtins
import io
import os
from pathlib import Path
import runpy
import sys
import types

import pytest

SETUP = Path(__file__).resolve().parents[1] / 'setup.py'

@pytest.mark.parametrize('platform', ['linux', 'win32'])
@pytest.mark.parametrize('architecture', [None, '7.5;8.6+PTX'])
def test_build_preserves_environment_and_uses_platform_flags(monkeypatch, platform, architecture):
    captured = {}
    extension = types.ModuleType('torch.utils.cpp_extension')
    extension.BuildExtension = type('BuildExtension', (), {})
    extension.CUDAExtension = lambda **kwargs: captured.setdefault('extension', kwargs)
    setuptools = types.ModuleType('setuptools')
    setuptools.find_packages = lambda **kwargs: ['neurosim_cu_esim']
    setuptools.setup = lambda **kwargs: captured.setdefault('setup', kwargs)
    monkeypatch.setitem(sys.modules, 'setuptools', setuptools)
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    monkeypatch.setitem(sys.modules, 'torch.utils', types.ModuleType('torch.utils'))
    monkeypatch.setitem(sys.modules, 'torch.utils.cpp_extension', extension)
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setenv('CC', 'user-selected-cc')
    monkeypatch.setenv('CXX', 'user-selected-cxx')
    if architecture is None:
        monkeypatch.delenv('TORCH_CUDA_ARCH_LIST', raising=False)
    else:
        monkeypatch.setenv('TORCH_CUDA_ARCH_LIST', architecture)
    before = dict(os.environ)
    old_open, old_io_open = builtins.open, io.open
    def read_only(delegate):
        def checked(file, mode='r', *args, **kwargs):
            assert not any(flag in mode for flag in ('w','a','+','x')), 'build config must not write files'
            return delegate(file, mode, *args, **kwargs)
        return checked
    monkeypatch.setattr(builtins, 'open', read_only(old_open))
    monkeypatch.setattr(io, 'open', read_only(old_io_open))
    values = runpy.run_path(str(SETUP), run_name='__main__')
    assert dict(os.environ) == before
    flags = captured['extension']['extra_compile_args']
    assert not any('gencode' in flag or 'arch=' in flag for flag in flags['nvcc'])
    assert '-fpermissive' not in flags['nvcc']
    assert '--use_fast_math' not in flags['nvcc']
    if platform == 'win32':
        assert '/O2' in flags['cxx'] and '/std:c++17' in flags['cxx']
        assert '-fPIC' not in flags['cxx'] + flags['nvcc']
    else:
        assert '-O3' in flags['cxx'] and '-fPIC' in flags['cxx']
        assert '/EHsc' not in flags['cxx'] + flags['nvcc']
    assert captured['setup']['cmdclass']['build_ext'] is extension.BuildExtension
    assert '_patch_pytorch_list_inl_header' not in values
    sources = captured['extension']['sources']
    assert any(str(s).endswith('evsim_graca_kernel.cu') for s in sources)
    assert len(sources) == 4

# use cuda extenstions to build my activations.cu file
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_activations',
    ext_modules=[
        CUDAExtension('my_activations', [
            'activations.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )

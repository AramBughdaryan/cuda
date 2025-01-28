from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='increment_extension',
    ext_modules=[
        CUDAExtension(
            name='increment_extension',
            sources=['increment.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
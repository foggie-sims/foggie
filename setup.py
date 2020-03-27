from setuptools import setup
#testcomment
setup(name="foggie",
      version="0.0.0",
      description="FOGGIE analysis tools",
      author="the FOGGIE team",
      author_email="molly@stsci.edu",
      license="BSD",
      keywords=["simulation", "astronomy", "astrophysics"],
      url="https://github.com/foggie-sims/foggie",
      packages=["foggie"],
      include_package_data=True,
      classifiers=[
          "Development Status :: 1 - Planning",
          "Environment :: Console",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          "License :: OSI Approved :: BSD License",
          "Operating System :: MacOS :: MacOS X",
          "Operating System :: POSIX :: Linux",
          "Operating System :: Unix",
          "Natural Language :: English",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
      ],
      install_requires=[
          'numpy',
          'yt>=3.4',
          'astropy',
          'trident',
          'datashader',
          'seaborn'
      ],
)

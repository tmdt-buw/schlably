pushd %~dp0

# Command file for Sphinx documentation

if [ "$SPHINXBUILD" = "" ]; then
	export SPHINXBUILD=sphinx-build
fi

export SOURCEDIR=source
export BUILDDIR=build

$SPHINXBUILD >NUL 2>NUL

$SPHINXBUILD -M $1 $SOURCEDIR $BUILDDIR $SPHINXOPTS $O

%SPHINXBUILD% -M help $SOURCEDIR $BUILDDIR $SPHINXOPTS $O

:end
popd

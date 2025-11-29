#!/usr/bin/ksh
# =========================================
# AIX Package Build Script for NeomindAI
# Auto-fills pkginfo and temp directories
# =========================================

set -e

# ------------------------------
# 1️⃣ Arguments
# ------------------------------
TEMPDIR=$1
if [ -z "$TEMPDIR" ]; then
    echo "Usage: $0 <tempdir>" 1>&2
    exit 1
fi

BUILD=`pwd`

# ------------------------------
# 2️⃣ Auto-fill pkginfo if missing
# ------------------------------
PKGINFO_FILE="$BUILD/build/aix/pkginfo"
mkdir -p "$BUILD/build/aix"
if [ ! -f "$PKGINFO_FILE" ]; then
cat > "$PKGINFO_FILE" <<EOF
PKG="neomind"
NAME="NeomindAI"
VERSION="1.0.0"
VENDOR="QUBUHUB"
ARCH="$(uname -m)"
EOF
    echo "Auto-created pkginfo at $PKGINFO_FILE"
fi

. "$PKGINFO_FILE"

# ------------------------------
# 3️⃣ Auto-populate temp directory
# ------------------------------
for dir in etc opt var httpd-root; do
    TARGET="$TEMPDIR/$dir/$NAME"
    if [ ! -d "$TARGET" ]; then
        echo "Creating directory $TARGET"
        mkdir -p "$TARGET"
        # Add dummy content for testing
        echo "Dummy file for $dir" > "$TARGET/dummy.txt"
    fi
done

# Add dummy man page
MAN_DIR="$TEMPDIR/usr/share/man"
mkdir -p "$MAN_DIR"
echo "NeomindAI manual" > "$MAN_DIR/neomind.1"

# ------------------------------
# 4️⃣ Prepare info template
# ------------------------------
INFO=$BUILD/build/aix/.info
mkdir -p "$INFO"
template=${INFO}/${PKG}.${NAME}.${VERSION}.template
> $template

# ------------------------------
# 5️⃣ Calculate folder sizes
# ------------------------------
cd ${TEMPDIR}
for d in etc opt var; do
    set `du -s $d/${NAME}`
    eval sz$d=$1+1
done

set `du -s usr/share/man`
szman=$1+1

# ------------------------------
# 6️⃣ Set permissions & ownership
# ------------------------------
files=./httpd-root
find ${files} -type d -exec chmod og+rx {} \;
chmod -R go+r ${files}
chown -R 0:0 ${files}

# ------------------------------
# 7️⃣ Create template file
# ------------------------------
cat <<EOF >>$template
Package Name: ${PKG}.${NAME}
Package VRMF: ${VERSION}.0
Update: N
Fileset
  Fileset Name: ${PKG}.${NAME}.rte
  Fileset VRMF: ${VERSION}.0
  Fileset Description: ${VENDOR} ${NAME} for ${ARCH}
  USRLIBLPPFiles
  EOUSRLIBLPPFiles
  Bosboot required: N
  License agreement acceptance required: N
  Include license files in this package: N
  Requisites:
        Upsize: /usr/share/man ${szman};
        Upsize: /etc/${NAME} $szetc;
        Upsize: /opt/${NAME} $szopt;
        Upsize: /var/${NAME} $szvar;
  USRFiles
EOF

find ${files} | sed -e s#^${files}## | sed -e "/^$/d" >>$template

cat <<EOF >>$template
  EOUSRFiles
  ROOT Part: N
  ROOTFiles
  EOROOTFiles
  Relocatable: N
EOFileset
EOF

cp ${template} ${BUILD}/build/aix

# ------------------------------
# 8️⃣ Build package using mkinstallp
# ------------------------------
mkinstallp -d ${TEMPDIR} -T ${template}

cp ${TEMPDIR}/tmp/$PKG.$NAME.$VERSION.0.bff ${BUILD}/build/aix
cd $BUILD/build/aix
rm -f $PKG.$NAME.$VERSION.$ARCH.I
mv $PKG.$NAME.$VERSION.0.bff $PKG.$NAME.$VERSION.$ARCH.I
rm .toc
inutoc .
installp -d . -ap ${PKG}.${NAME}
installp -d . -L

echo "✅ Package build complete: build/aix/${PKG}.${NAME}.${VERSION}.${ARCH}.I"

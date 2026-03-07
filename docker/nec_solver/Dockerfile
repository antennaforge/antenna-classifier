# Use Debian slim for easy apt packages (nec2++).
# On Apple Silicon (M1/M2), you can build/run with --platform=linux/amd64 if needed.
FROM python:3.12-slim

# System deps + NEC solver
RUN apt-get update && apt-get install -y --no-install-recommends \
    nec2c xnec2c ca-certificates build-essential git automake autoconf libtool wget && \
    mkdir -p /tmp/build && cd /tmp/build && \
    git clone --depth=1 https://github.com/tmolteno/necpp.git . && \
    make -f Makefile.git > /tmp/necpp.makegit.log 2>&1 && ./configure --without-lapack > /tmp/necpp.configure.log 2>&1 && \
    sed -i 's/^nec2___LDFLAGS = -all-static/nec2___LDFLAGS = /' src/Makefile || true && \
    make -j"$(nproc)" V=1 > /tmp/necpp.build.log 2>&1 || (echo 'Build failed; tail build log:' && tail -n 200 /tmp/necpp.build.log && false) && \
    make install > /tmp/necpp.install.log 2>&1 || (echo 'Install failed; tail log:' && tail -n 200 /tmp/necpp.install.log && false) && \
    mkdir -p /usr/local/share/necpp-build && cp /tmp/necpp.*.log /usr/local/share/necpp-build/ || true && \
    ls -l /usr/local/bin && (command -v nec2++ || echo 'nec2++ NOT FOUND') && \
    rm -rf /var/lib/apt/lists/*

# App files
WORKDIR /app
COPY api.py /app/api.py
RUN pip install --no-cache-dir fastapi uvicorn

# Non-root user
RUN useradd -m solver && mkdir -p /cache && chown -R solver:solver /app /cache
USER solver

ENV PORT=8787
# Default solver binary; override with: docker run -e NEC_BIN=nec2++ ...
ENV NEC_BIN=nec2c
EXPOSE 8787
CMD ["uvicorn","api:app","--host","0.0.0.0","--port","8787","--workers","1"]

FROM bitnami/spark:3.5
USER root
RUN install_packages python3 python3-distutils python3-minimal || \
    (apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*)
USER 0


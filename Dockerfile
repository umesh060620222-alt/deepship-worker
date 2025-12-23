# Uses your pre-built base image with all dependencies
FROM umesh685/deepship-base:latest

WORKDIR /app

# Copy application code (dependencies already in base image)
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start API
CMD ["python", "worker.py"]
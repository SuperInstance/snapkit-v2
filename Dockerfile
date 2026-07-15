-- snapkit-v2 official Docker image
-- Runs the Fleet Coordinator with the web dashboard on port 8000

FROM python:3.11-slim

WORKDIR /app

# Install package
COPY . /app
RUN pip install --no-cache-dir -e .

# Expose the dashboard port
EXPOSE 8000

# Default: run the harmony dashboard
# Override with: docker run -p 8000:8000 snapkit snapkit harmony --port 8000
CMD ["snapkit", "info"]
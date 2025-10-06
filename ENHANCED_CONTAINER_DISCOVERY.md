# Enhanced Container Discovery System

## 🚀 **Overview**

The enhanced container discovery system provides intelligent, multi-layered detection of frontend and backend containers in your projects. It automatically classifies containers using sophisticated heuristics and maintains backward compatibility with existing file-based discovery.

## 🎯 **Key Features**

### **1. Multi-Strategy Discovery**
- **Docker Compose Detection** - Parses compose files for service definitions
- **Running Container Analysis** - Queries active Docker containers
- **Framework Detection** - Identifies project frameworks from dependency files
- **Enhanced File Discovery** - Improved path-based classification (fallback)

### **2. Intelligent Classification**
- **Service Name Analysis** - Keywords like "frontend", "backend", "api", "web"
- **Port-Based Classification** - Common ports (3000, 5173 = frontend; 5000, 8000 = backend)
- **Framework Detection** - React, Vue, Django, Flask, Express, etc.
- **Image Analysis** - Docker image names and tags

### **3. Framework Support**
- **Frontend**: React, Vue, Angular, Next.js, Nuxt.js, Vite
- **Backend**: Django, Flask, FastAPI, Express, Node.js, Go, Rust, Java
- **Database**: MongoDB, PostgreSQL, Redis, MySQL

## 🔧 **How It Works**

### **Discovery Strategies (in order)**

1. **Docker Compose Services**
   ```python
   # Scans for compose files
   docker-compose.yml, docker-compose.yaml, compose.yml, compose.yaml
   
   # Analyzes service definitions
   services:
     frontend:
       ports: ["3000:3000"]
     backend:
       ports: ["5000:5000"]
   ```

2. **Running Container Detection**
   ```python
   # Queries running containers
   docker ps --format json
   
   # Classifies by name, image, ports
   - kyc-frontend-test → frontend
   - kyc-backend → backend
   - kyc-mongodb-test → app
   ```

3. **Framework Detection**
   ```python
   # Frontend indicators
   package.json, yarn.lock, vite.config.js, next.config.js
   
   # Backend indicators  
   requirements.txt, Cargo.toml, go.mod, pom.xml
   ```

4. **Enhanced File Discovery**
   ```python
   # Improved path classification
   logs/frontend/ → frontend
   logs/backend/ → backend
   src/app/ → backend
   public/ → frontend
   ```

## 📊 **Classification Logic**

### **Frontend Indicators**
- **Keywords**: frontend, front, web, ui, client, react, vue, angular
- **Ports**: 3000, 5173, 8080, 4200
- **Frameworks**: React, Vue, Angular, Next.js, Nuxt.js
- **Paths**: public/, static/, src/components/

### **Backend Indicators**
- **Keywords**: backend, back, api, server, app, django, flask, express
- **Ports**: 5000, 8000, 9000, 3001
- **Frameworks**: Django, Flask, FastAPI, Express, Node.js
- **Paths**: src/, api/, server/, app/

### **Default Classification**
- **App**: Database containers, utilities, unknown services

## 🎨 **Usage Examples**

### **Docker Compose Project**
```yaml
# docker-compose.yml
services:
  web:
    build: ./frontend
    ports: ["3000:3000"]
  api:
    build: ./backend  
    ports: ["5000:5000"]
  db:
    image: postgres:13
```

**Result**: `web` → frontend, `api` → backend, `db` → app

### **Running Containers**
```bash
# Container names
kyc-frontend-test    → frontend
kyc-backend         → backend  
kyc-mongodb-test    → app
```

### **Framework Detection**
```bash
# Project structure
package.json        → Frontend framework detected
requirements.txt    → Backend framework detected
```

## 🔍 **Test Results**

The enhanced discovery successfully detected:

```
📋 Found 3 container(s):
Container: frontend
Service: kyc-frontend-test
Log File: docker_logs_kyc-frontend-test.log

Container: backend  
Service: kyc-backend
Log File: docker_logs_kyc-backend.log

Container: app
Service: kyc-mongodb-test
Log File: docker_logs_kyc-mongodb-test.log
```

## 🚀 **Benefits**

### **1. Automatic Classification**
- No manual configuration required
- Intelligent heuristics for accurate classification
- Handles edge cases and unknown services

### **2. Multi-Project Support**
- Works with any project structure
- Supports multiple frameworks simultaneously
- Handles complex microservice architectures

### **3. Backward Compatibility**
- Maintains existing file-based discovery
- Graceful fallback to simple path matching
- No breaking changes to existing functionality

### **4. Extensible Design**
- Easy to add new classification rules
- Pluggable discovery strategies
- Environment variable overrides

## 🛠 **Configuration**

### **Environment Variables**
```bash
# Override Docker containers
DSPY_DOCKER_CONTAINERS="frontend,backend,api"

# Additional vector topics
DSPY_VECTOR_TOPICS="custom.topic,another.topic"
```

### **Manual Override**
```python
# Force specific container types
discoveries = autodiscover_logs(workspace)
# Manually adjust classifications if needed
```

## 🔧 **Advanced Features**

### **Service Label Detection**
```yaml
# docker-compose.yml with labels
services:
  web:
    labels:
      - "dspy.type=frontend"
      - "dspy.service=react-app"
```

### **Port-Based Classification**
```python
# Automatic port detection
ports: ["3000:3000"] → frontend
ports: ["5000:5000"] → backend
ports: ["5432:5432"] → database
```

### **Framework Auto-Detection**
```python
# Detects frameworks from project files
package.json → Node.js/React
requirements.txt → Python/Django
Cargo.toml → Rust
go.mod → Go
```

## 📈 **Performance**

- **Fast Discovery**: Multi-threaded container detection
- **Efficient Classification**: O(1) keyword matching
- **Minimal Overhead**: Lazy evaluation of discovery strategies
- **Caching**: Reuses container information when possible

## 🎯 **Future Enhancements**

1. **Kubernetes Support** - Pod and service detection
2. **Service Mesh Integration** - Istio, Linkerd support
3. **Cloud Provider Detection** - AWS, GCP, Azure services
4. **Machine Learning Classification** - AI-powered container classification
5. **Real-time Monitoring** - Live container state tracking

## 🚀 **Getting Started**

The enhanced discovery is automatically enabled when you run:

```bash
# Start the agent with enhanced discovery
dspy-agent up --workspace /path/to/project

# Or use the validation script
./validate_streaming_pipeline.sh
```

The system will automatically detect and classify your frontend and backend containers, providing intelligent log streaming and monitoring for your entire development stack!

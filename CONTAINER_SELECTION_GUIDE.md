# Docker Container Selection Guide

## 🎯 **Overview**

The DSPy agent now includes an interactive container selection system that allows you to choose which Docker containers to monitor for logs and streaming. This gives you full control over which frontend and backend containers are included in your development workflow.

## 🚀 **Quick Start**

### **1. List Available Containers**
```bash
# See all available Docker containers
dspy-agent select-containers --list
```

### **2. Interactive Selection**
```bash
# Interactive container selection
dspy-agent select-containers
```

### **3. Start Agent with Selected Containers**
```bash
# The selection command will show you the environment variable
export DSPY_DOCKER_CONTAINERS=kyc-frontend-test,kyc-backend,kyc-mongodb-test
dspy-agent up
```

## 🔍 **Container Discovery**

The system automatically discovers and classifies containers:

### **Frontend Containers**
- **Keywords**: frontend, front, web, ui, client, react, vue, angular
- **Ports**: 3000, 5173, 8080, 4200
- **Examples**: React apps, Vue apps, Next.js, Nuxt.js

### **Backend Containers**
- **Keywords**: backend, back, api, server, app, django, flask, express
- **Ports**: 5000, 8000, 9000, 3001
- **Examples**: Django, Flask, FastAPI, Express, Node.js APIs

### **Database Containers**
- **Keywords**: mysql, postgres, mongodb, redis, elasticsearch
- **Ports**: 5432, 3306, 6379
- **Examples**: PostgreSQL, MySQL, MongoDB, Redis

## 🎨 **Selection Methods**

### **1. Interactive Selection**
```bash
dspy-agent select-containers
```
- Shows numbered list of containers
- Select by number: `1,3,5`
- Select by type: `frontend`, `backend`, `database`
- Select all running: `all`

### **2. List Only**
```bash
dspy-agent select-containers --list
```
- Shows all containers without selection
- Displays classification summary
- Useful for exploring available containers

### **3. Manual Environment Variable**
```bash
# Set containers manually
export DSPY_DOCKER_CONTAINERS="frontend-app,backend-api,database"
dspy-agent up
```

## 📊 **Example Output**

```
🐳 Available Docker Containers
========================================

#   Name                      Image                          Status          Type      
-------------------------------------------------------------------------------------
1   kyc-frontend-test         monerisapplication-frontend-t  🟢 Up 23 minutes  frontend  
2   kyc-backend               monerisapplication-backend-te  🟢 Up 24 minutes  backend   
3   kyc-mongodb-test          mongo:8                        🟢 Up 24 minutes  database  
4   lightweight-dspy-agent-1  dspy-lightweight:latest        🟢 Up 44 minutes unknown   
5   lightweight-ollama-1      ollama/ollama:latest           🟢 Up 44 minutes  unknown   

📊 Container Summary:
  frontend: 5 containers
  backend: 6 containers
  database: 2 containers
  unknown: 15 containers
```

## 🎯 **Selection Examples**

### **Select by Type**
```
Selection: frontend
# Selects all frontend containers
```

### **Select by Number**
```
Selection: 1,3,5
# Selects containers 1, 3, and 5
```

### **Select All Running**
```
Selection: all
# Selects all running containers
```

### **Select Specific Containers**
```
Selection: kyc-frontend-test,kyc-backend
# Selects specific containers by name
```

## 🔧 **Configuration**

### **Saved Configuration**
The selection is automatically saved to `.dspy_selected_containers.json`:

```json
{
  "selected_containers": [
    "kyc-frontend-test",
    "kyc-backend", 
    "kyc-mongodb-test"
  ],
  "workspace": "/path/to/workspace",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### **Environment Variables**
```bash
# Set selected containers
export DSPY_DOCKER_CONTAINERS="frontend-app,backend-api"

# Additional vector topics
export DSPY_VECTOR_TOPICS="custom.topic,another.topic"

# Start agent
dspy-agent up
```

## 🚀 **Integration with Agent**

### **Automatic Discovery**
The enhanced `autodiscover_logs()` function now:
1. **Checks saved configuration** - Uses previously selected containers
2. **Discovers running containers** - Queries `docker ps` for active containers
3. **Classifies intelligently** - Uses name, image, and port analysis
4. **Falls back gracefully** - Uses file-based discovery if needed

### **Streaming Integration**
Selected containers are automatically:
- **Monitored for logs** - Real-time log streaming
- **Classified by type** - Frontend/backend/database topics
- **Included in vectorization** - Features extracted for RL training
- **Tracked in metrics** - Performance and health monitoring

## 🎨 **Advanced Usage**

### **Custom Classification**
You can influence classification by:
- **Container naming** - Use `frontend-`, `backend-`, `api-` prefixes
- **Port mapping** - Expose standard ports (3000, 5000, 8000)
- **Image tags** - Use descriptive image names
- **Labels** - Add Docker labels for classification

### **Multiple Projects**
```bash
# Project A
cd /path/to/project-a
dspy-agent select-containers
export DSPY_DOCKER_CONTAINERS="project-a-frontend,project-a-backend"

# Project B  
cd /path/to/project-b
dspy-agent select-containers
export DSPY_DOCKER_CONTAINERS="project-b-frontend,project-b-api"
```

### **Development Workflow**
```bash
# 1. Start your application containers
docker-compose up -d

# 2. Select containers to monitor
dspy-agent select-containers

# 3. Start DSPy agent with selected containers
export DSPY_DOCKER_CONTAINERS="your-selected-containers"
dspy-agent up

# 4. Agent now monitors your application logs!
```

## 🔍 **Troubleshooting**

### **No Containers Found**
```bash
# Check Docker is running
docker ps

# Check container names
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
```

### **Classification Issues**
```bash
# Check container details
docker inspect <container-name>

# Manual classification
export DSPY_DOCKER_CONTAINERS="manually-classified-containers"
```

### **Selection Not Working**
```bash
# Clear saved configuration
rm .dspy_selected_containers.json

# Re-run selection
dspy-agent select-containers
```

## 🎯 **Best Practices**

### **1. Use Descriptive Names**
```bash
# Good
frontend-react-app
backend-django-api
database-postgres

# Avoid
container1, container2, container3
```

### **2. Standard Ports**
```yaml
# docker-compose.yml
services:
  frontend:
    ports: ["3000:3000"]  # Standard frontend port
  backend:
    ports: ["5000:5000"]  # Standard backend port
```

### **3. Regular Updates**
```bash
# Update selection when containers change
dspy-agent select-containers --list
dspy-agent select-containers
```

### **4. Environment Management**
```bash
# Use .env files for different environments
echo "DSPY_DOCKER_CONTAINERS=dev-frontend,dev-backend" > .env.dev
echo "DSPY_DOCKER_CONTAINERS=prod-frontend,prod-backend" > .env.prod
```

## 🚀 **Getting Started**

1. **Start your application containers**
2. **Run container selection**: `dspy-agent select-containers`
3. **Copy the environment variable** shown in the output
4. **Start the agent**: `export DSPY_DOCKER_CONTAINERS="..." && dspy-agent up`
5. **Monitor your application logs** in the DSPy dashboard!

The agent will now intelligently monitor your selected frontend and backend containers, providing real-time log streaming and intelligent analysis for your development workflow! 🎉

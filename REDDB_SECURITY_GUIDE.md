# RedDB Security Guide

## 🔒 Security Features Implemented

### 1. Authentication & Authorization
- **Bearer Token Authentication**: All API calls require a valid Bearer token
- **Secure Token Generation**: 64-character hex tokens generated with `openssl rand -hex 32`
- **Token Validation**: Scripts validate token format before use
- **Token Masking**: Tokens are masked in output for security

### 2. Network Security
- **Localhost-Only Binding**: RedDB bound to `127.0.0.1:8080` (not `0.0.0.0`)
- **Internal Docker Network**: Isolated network with no external access
- **Rate Limiting**: Nginx proxy with rate limiting (10 requests/second)
- **Security Headers**: X-Frame-Options, X-Content-Type-Options, etc.

### 3. Container Security
- **No New Privileges**: `no-new-privileges:true`
- **Read-Only Filesystem**: Container filesystem is read-only
- **Dropped Capabilities**: All capabilities dropped except `NET_BIND_SERVICE`
- **Temporary Filesystems**: Secure tmpfs mounts for temporary data

### 4. Environment Security
- **Secure Environment Variables**: Tokens never logged in plaintext
- **Validation**: All environment variables validated before use
- **Warnings**: Clear security warnings for missing tokens

## 🚀 Quick Start (Secure)

### Option 1: Basic Secure Setup
```bash
# Set up environment
./setup_reddb_env.sh

# Start RedDB with security
./start_reddb.sh
```

### Option 2: Ultra-Secure Setup (Recommended)
```bash
# Set up environment
./setup_reddb_env.sh

# Start with maximum security (Docker Compose + Nginx)
./start_reddb_secure.sh
```

## 🔧 Configuration Files

### Environment Setup
- `setup_reddb_env.sh` - Sets up secure environment variables
- `start_reddb.sh` - Basic secure startup script
- `start_reddb_secure.sh` - Ultra-secure startup with Docker Compose

### Docker Configuration
- `docker/reddb-secure.yml` - Secure Docker Compose configuration
- `docker/nginx-reddb.conf` - Nginx proxy with security headers

## 🛡️ Security Best Practices

### 1. Token Management
```bash
# Generate secure token
export REDDB_ADMIN_TOKEN=$(openssl rand -hex 32)

# Save to shell profile (secure)
echo 'export REDDB_ADMIN_TOKEN=your_token_here' >> ~/.zshrc
```

### 2. Environment Variables
```bash
# Required secure variables
export REDDB_URL=http://127.0.0.1:8080
export REDDB_NAMESPACE=agent
export REDDB_TOKEN=$REDDB_ADMIN_TOKEN
export DB_BACKEND=reddb
```

### 3. Testing with Authentication
```bash
# All API calls require Bearer token
curl -X POST http://127.0.0.1:8767/api/db/ingest \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $REDDB_ADMIN_TOKEN" \
  -d '{"kind":"document","namespace":"agent","collection":"notes","id":"test1","text":"Your text"}'
```

## ⚠️ Security Warnings

### 1. Never Commit Tokens
- **NEVER** commit `REDDB_ADMIN_TOKEN` to version control
- Use `.env` files (not tracked) or environment variables
- Rotate tokens regularly in production

### 2. Network Access
- RedDB is bound to localhost only (`127.0.0.1`)
- No external network access by default
- Use VPN or SSH tunnels for remote access

### 3. Production Deployment
- Use proper secrets management (AWS Secrets Manager, HashiCorp Vault)
- Enable TLS/SSL for all connections
- Use strong, unique tokens per environment
- Monitor access logs regularly

## 🔍 Troubleshooting

### Health Checks
```bash
# Check RedDB health
curl -H "Authorization: Bearer $REDDB_ADMIN_TOKEN" http://127.0.0.1:8080/health

# Check agent backend
curl http://127.0.0.1:8767/api/db/health
```

### Logs
```bash
# RedDB logs
docker logs redb-open

# Docker Compose logs
docker-compose -f docker/reddb-secure.yml logs

# Agent backend logs
ps aux | grep databackend-fastapi
```

### Common Issues
1. **401 Unauthorized**: Check `REDDB_ADMIN_TOKEN` is set correctly
2. **Connection Refused**: Ensure RedDB is running and bound to localhost
3. **Token Invalid**: Regenerate with `openssl rand -hex 32`

## 🚨 Emergency Procedures

### Stop All Services
```bash
# Basic setup
docker stop redb-open && kill $BACKEND_PID

# Docker Compose setup
docker-compose -f docker/reddb-secure.yml down
```

### Rotate Tokens
```bash
# Generate new token
export REDDB_ADMIN_TOKEN=$(openssl rand -hex 32)

# Restart services with new token
./start_reddb_secure.sh
```

### Reset Everything
```bash
# Stop all services
docker-compose -f docker/reddb-secure.yml down
docker system prune -f

# Remove volumes (WARNING: Data loss!)
docker volume rm docker_reddb_data
```

## 📊 Security Monitoring

### Check Running Services
```bash
# Docker containers
docker ps | grep redb

# Network connections
netstat -tlnp | grep :8080
netstat -tlnp | grep :8767
```

### Verify Security
```bash
# Check localhost binding
ss -tlnp | grep :8080

# Test external access (should fail)
curl http://0.0.0.0:8080/health
```

## 🔐 Production Security Checklist

- [ ] Strong, unique tokens per environment
- [ ] Secrets management system in place
- [ ] TLS/SSL enabled for all connections
- [ ] Network access controls configured
- [ ] Monitoring and alerting set up
- [ ] Regular security audits scheduled
- [ ] Backup and recovery procedures tested
- [ ] Incident response plan documented

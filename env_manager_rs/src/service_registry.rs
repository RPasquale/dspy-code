use bollard::models::PortBinding;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ServiceDefinition {
    pub name: String,
    pub image: String,
    pub ports: HashMap<String, Vec<PortBinding>>,
    pub environment: Vec<String>,
    pub volumes: Vec<String>,
    pub depends_on: Vec<String>,
    pub health_check_url: Option<String>,
    #[allow(dead_code)]
    pub health_check_interval_secs: u64,
    pub required: bool,
    pub network: Option<String>,
}

pub struct ServiceRegistry {
    services: HashMap<String, ServiceDefinition>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            services: HashMap::new(),
        };
        registry.register_default_services();
        registry
    }

    fn register_default_services(&mut self) {
        // Redis - Cache and pub/sub
        self.register(ServiceDefinition {
            name: "redis".to_string(),
            image: "redis:7-alpine".to_string(),
            ports: Self::create_port_binding("6379", "6379"),
            environment: vec![],
            volumes: vec!["redis_data:/data".to_string()],
            depends_on: vec![],
            health_check_url: Some("http://localhost:6379".to_string()), // Redis PING
            health_check_interval_secs: 2,
            required: true,
            network: Some("lightweight_default".to_string()),
        });

        // RedDB - Lightweight database
        self.register(ServiceDefinition {
            name: "redb".to_string(),
            image: "dspy-lightweight:latest".to_string(),
            ports: Self::create_port_binding("8080", "8082"),
            environment: vec![
                "REDB_DATA_DIR=/data".to_string(),
                "REDB_HOST=0.0.0.0".to_string(),
                "REDB_PORT=8080".to_string(),
                "REDB_NAMESPACE=dspy".to_string(),
            ],
            volumes: vec!["redb_data:/data".to_string()],
            depends_on: vec![],
            health_check_url: Some("http://localhost:8082/health".to_string()),
            health_check_interval_secs: 2,
            required: true,
            network: Some("lightweight_default".to_string()),
        });

        // Ollama - Local LLM runtime
        self.register(ServiceDefinition {
            name: "ollama".to_string(),
            image: "ollama/ollama:latest".to_string(),
            ports: Self::create_port_binding("11434", "11435"),
            environment: vec!["OLLAMA_HOST=0.0.0.0:11434".to_string()],
            volumes: vec!["ollama_data:/root/.ollama".to_string()],
            depends_on: vec![],
            health_check_url: Some("http://localhost:11435/api/tags".to_string()),
            health_check_interval_secs: 3,
            required: false,
            network: Some("lightweight_default".to_string()),
        });

        // InferMesh Node A - Inference service
        self.register(ServiceDefinition {
            name: "infermesh-node-a".to_string(),
            image: "official-infermesh:latest".to_string(),
            ports: HashMap::new(), // Internal network only
            environment: vec![
                "MESH_NODE_ID=node-a".to_string(),
                "MESH_PORT=9000".to_string(),
                "REDIS_URL=redis://redis:6379".to_string(),
            ],
            volumes: vec!["infermesh_cache_a:/cache".to_string()],
            depends_on: vec!["redis".to_string()],
            health_check_url: None, // Container health check only
            health_check_interval_secs: 3,
            required: false,
            network: Some("lightweight_default".to_string()),
        });

        // InferMesh Node B
        self.register(ServiceDefinition {
            name: "infermesh-node-b".to_string(),
            image: "official-infermesh:latest".to_string(),
            ports: HashMap::new(), // Internal network only
            environment: vec![
                "MESH_NODE_ID=node-b".to_string(),
                "MESH_PORT=9000".to_string(),
                "REDIS_URL=redis://redis:6379".to_string(),
            ],
            volumes: vec!["infermesh_cache_b:/cache".to_string()],
            depends_on: vec!["redis".to_string()],
            health_check_url: None, // Container health check only
            health_check_interval_secs: 3,
            required: false,
            network: Some("lightweight_default".to_string()),
        });

        // InferMesh Router
        self.register(ServiceDefinition {
            name: "infermesh-router".to_string(),
            image: "official-infermesh:latest".to_string(),
            ports: Self::create_port_binding("9000", "19000"),
            environment: vec![
                "MESH_ROLE=router".to_string(),
                "MESH_PORT=9000".to_string(),
                "MESH_NODES=infermesh-node-a:9000,infermesh-node-b:9000".to_string(),
                "REDIS_URL=redis://redis:6379".to_string(),
            ],
            volumes: vec![],
            depends_on: vec![
                "infermesh-node-a".to_string(),
                "infermesh-node-b".to_string(),
            ],
            health_check_url: Some("http://localhost:19000/health".to_string()),
            health_check_interval_secs: 3,
            required: false,
            network: Some("lightweight_default".to_string()),
        });

        // Zookeeper - For Kafka
        self.register(ServiceDefinition {
            name: "zookeeper".to_string(),
            image: "confluentinc/cp-zookeeper:7.5.0".to_string(),
            ports: Self::create_port_binding("2181", "2181"),
            environment: vec![
                "ZOOKEEPER_CLIENT_PORT=2181".to_string(),
                "ZOOKEEPER_TICK_TIME=2000".to_string(),
            ],
            volumes: vec!["zookeeper_data:/var/lib/zookeeper/data".to_string()],
            depends_on: vec![],
            health_check_url: None,
            health_check_interval_secs: 5,
            required: false,
            network: Some("lightweight_default".to_string()),
        });

        // Kafka - Event streaming
        self.register(ServiceDefinition {
            name: "kafka".to_string(),
            image: "confluentinc/cp-kafka:7.5.0".to_string(),
            ports: Self::create_port_binding("29092", "9092"),
            environment: vec![
                "KAFKA_BROKER_ID=1".to_string(),
                "KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181".to_string(),
                "KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092".to_string(),
                "KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT".to_string(),
                "KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT".to_string(),
                "KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1".to_string(),
            ],
            volumes: vec!["kafka_data:/var/lib/kafka/data".to_string()],
            depends_on: vec!["zookeeper".to_string()],
            health_check_url: None,
            health_check_interval_secs: 10,
            required: false,
            network: Some("lightweight_default".to_string()),
        });

        // Prometheus - Metrics collection
        self.register(ServiceDefinition {
            name: "prometheus".to_string(),
            image: "prom/prometheus:latest".to_string(),
            ports: Self::create_port_binding("9090", "9090"),
            environment: vec![],
            volumes: vec![
                "prometheus_data:/prometheus".to_string(),
                "./docker/lightweight/prometheus.yml:/etc/prometheus/prometheus.yml:ro".to_string(),
            ],
            depends_on: vec![],
            health_check_url: Some("http://localhost:9090/-/healthy".to_string()),
            health_check_interval_secs: 5,
            required: false,
            network: Some("lightweight_default".to_string()),
        });

        // Go Orchestrator - Started separately by dspy-agent CLI, not as container
        // self.register(ServiceDefinition {
        //     name: "go-orchestrator".to_string(),
        //     image: "dspy-lightweight:latest".to_string(),
        //     ports: Self::create_port_binding("9097", "9097"),
        //     environment: vec![
        //         "KAFKA_BROKERS=kafka:29092".to_string(),
        //         "REDIS_URL=redis://redis:6379".to_string(),
        //     ],
        //     volumes: vec![],
        //     depends_on: vec!["kafka".to_string(), "redis".to_string()],
        //     health_check_url: Some("http://localhost:9097/health".to_string()),
        //     health_check_interval_secs: 3,
        //     required: true,
        //     network: Some("lightweight_default".to_string()),
        // });
    }

    fn create_port_binding(
        container_port: &str,
        host_port: &str,
    ) -> HashMap<String, Vec<PortBinding>> {
        let mut ports = HashMap::new();
        ports.insert(
            format!("{}/tcp", container_port),
            vec![PortBinding {
                host_ip: Some("127.0.0.1".to_string()),
                host_port: Some(host_port.to_string()),
            }],
        );
        ports
    }

    pub fn register(&mut self, service: ServiceDefinition) {
        self.services.insert(service.name.clone(), service);
    }

    pub fn get(&self, name: &str) -> Option<&ServiceDefinition> {
        self.services.get(name)
    }

    pub fn get_all(&self) -> Vec<&ServiceDefinition> {
        self.services.values().collect()
    }

    #[allow(dead_code)]
    pub fn get_required(&self) -> Vec<&ServiceDefinition> {
        self.services.values().filter(|s| s.required).collect()
    }

    /// Get services in dependency order (dependencies first)
    pub fn get_startup_order(&self) -> Vec<&ServiceDefinition> {
        let mut ordered = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn visit<'a>(
            service: &'a ServiceDefinition,
            registry: &'a HashMap<String, ServiceDefinition>,
            visited: &mut std::collections::HashSet<String>,
            ordered: &mut Vec<&'a ServiceDefinition>,
        ) {
            if visited.contains(&service.name) {
                return;
            }

            visited.insert(service.name.clone());

            // Visit dependencies first
            for dep_name in &service.depends_on {
                if let Some(dep) = registry.get(dep_name) {
                    visit(dep, registry, visited, ordered);
                }
            }

            ordered.push(service);
        }

        for service in self.services.values() {
            visit(service, &self.services, &mut visited, &mut ordered);
        }

        ordered
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

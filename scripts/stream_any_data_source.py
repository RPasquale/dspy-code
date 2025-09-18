#!/usr/bin/env python3
"""
Stream Any Data Source - Universal Data Ingestion Interface

This script provides a simple interface to point the streaming engine at any data source
and automatically ingest, vectorize, and create training data from it.

Usage:
    # Stream from GitHub API
    python stream_any_data_source.py --github-repo microsoft/vscode --commits
    
    # Stream from a database
    python stream_any_data_source.py --database postgresql --query "SELECT * FROM users"
    
    # Stream from a file
    python stream_any_data_source.py --file /path/to/data.json
    
    # Stream from a webhook
    python stream_any_data_source.py --webhook --port 8080
"""

import os
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.enhanced_streaming_connectors import EnhancedStreamingOrchestrator, DataSourceConfig

class UniversalDataStreamer:
    """Universal interface for streaming any data source"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.orchestrator = EnhancedStreamingOrchestrator(project_root)
    
    def add_github_source(self, repo: str, data_type: str = "commits", poll_interval: float = 30.0):
        """Add GitHub repository as data source"""
        if data_type == "commits":
            url = f"https://api.github.com/repos/{repo}/commits"
        elif data_type == "issues":
            url = f"https://api.github.com/repos/{repo}/issues"
        elif data_type == "pulls":
            url = f"https://api.github.com/repos/{repo}/pulls"
        else:
            raise ValueError(f"Unsupported GitHub data type: {data_type}")
        
        config = DataSourceConfig(
            name=f"github_{repo.replace('/', '_')}_{data_type}",
            type="api",
            config={
                "url": url,
                "method": "GET",
                "headers": {"Accept": "application/vnd.github.v3+json"},
                "params": {"per_page": 10}
            },
            poll_interval=poll_interval
        )
        
        return self.orchestrator.add_data_source(config)
    
    def add_database_source(self, db_type: str, **kwargs):
        """Add database as data source"""
        config = DataSourceConfig(
            name=f"database_{db_type}_{kwargs.get('database', 'default')}",
            type="database",
            config={
                "type": db_type,
                **kwargs
            }
        )
        
        return self.orchestrator.add_data_source(config)
    
    def add_file_source(self, file_path: str, poll_interval: float = 5.0):
        """Add file as data source"""
        config = DataSourceConfig(
            name=f"file_{Path(file_path).stem}",
            type="file",
            config={
                "path": file_path
            },
            poll_interval=poll_interval
        )
        
        return self.orchestrator.add_data_source(config)
    
    def add_webhook_source(self, port: int = 8080, path: str = "/webhook"):
        """Add webhook as data source"""
        config = DataSourceConfig(
            name="webhook_receiver",
            type="webhook",
            config={
                "host": "localhost",
                "port": port,
                "path": path
            }
        )
        
        return self.orchestrator.add_data_source(config)
    
    def add_api_source(self, url: str, method: str = "GET", headers: Dict[str, str] = None, params: Dict[str, Any] = None):
        """Add generic API as data source"""
        config = DataSourceConfig(
            name=f"api_{url.split('//')[1].split('/')[0].replace('.', '_')}",
            type="api",
            config={
                "url": url,
                "method": method,
                "headers": headers or {},
                "params": params or {}
            }
        )
        
        return self.orchestrator.add_data_source(config)
    
    async def start_streaming(self):
        """Start streaming from all configured sources"""
        print("🚀 Starting Universal Data Streaming...")
        print("=" * 60)
        
        if not self.orchestrator.connectors:
            print("❌ No data sources configured!")
            return
        
        print(f"📊 Configured data sources:")
        for name, connector in self.orchestrator.connectors.items():
            print(f"  • {name} ({connector.source_config.type})")
        
        print("\n🔄 Starting data collection...")
        print("Press Ctrl+C to stop")
        
        await self.orchestrator.start()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Stream Any Data Source")
    
    # GitHub options
    parser.add_argument("--github-repo", help="GitHub repository (owner/repo)")
    parser.add_argument("--github-data", choices=["commits", "issues", "pulls"], default="commits", help="GitHub data type")
    parser.add_argument("--github-interval", type=float, default=30.0, help="GitHub polling interval (seconds)")
    
    # Database options
    parser.add_argument("--database", choices=["postgresql", "mysql", "mongodb"], help="Database type")
    parser.add_argument("--db-host", default="localhost", help="Database host")
    parser.add_argument("--db-port", type=int, help="Database port")
    parser.add_argument("--db-user", help="Database user")
    parser.add_argument("--db-password", help="Database password")
    parser.add_argument("--db-name", help="Database name")
    parser.add_argument("--db-query", help="Database query")
    parser.add_argument("--db-collection", help="MongoDB collection name")
    
    # File options
    parser.add_argument("--file", help="File path to stream")
    parser.add_argument("--file-interval", type=float, default=5.0, help="File polling interval (seconds)")
    
    # Webhook options
    parser.add_argument("--webhook", action="store_true", help="Start webhook receiver")
    parser.add_argument("--webhook-port", type=int, default=8080, help="Webhook port")
    parser.add_argument("--webhook-path", default="/webhook", help="Webhook path")
    
    # API options
    parser.add_argument("--api-url", help="API URL to stream")
    parser.add_argument("--api-method", default="GET", help="API method")
    parser.add_argument("--api-headers", help="API headers (JSON)")
    parser.add_argument("--api-params", help="API parameters (JSON)")
    
    # Configuration file
    parser.add_argument("--config", help="Configuration file (JSON)")
    
    args = parser.parse_args()
    
    # Create streamer
    streamer = UniversalDataStreamer(project_root)
    
    # Load configuration from file if provided
    if args.config:
        try:
            with open(args.config) as f:
                config_data = json.load(f)
            
            for source_config in config_data.get("sources", []):
                config = DataSourceConfig(**source_config)
                streamer.orchestrator.add_data_source(config)
                
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return
    
    # Add data sources based on command line arguments
    sources_added = 0
    
    # GitHub source
    if args.github_repo:
        if streamer.add_github_source(args.github_repo, args.github_data, args.github_interval):
            sources_added += 1
            print(f"✅ Added GitHub source: {args.github_repo} ({args.github_data})")
    
    # Database source
    if args.database:
        db_config = {
            "host": args.db_host,
            "user": args.db_user,
            "password": args.db_password,
            "database": args.db_name
        }
        
        if args.db_port:
            db_config["port"] = args.db_port
        
        if args.database == "mongodb":
            db_config["connection_string"] = f"mongodb://{args.db_user}:{args.db_password}@{args.db_host}:{args.db_port or 27017}/{args.db_name}"
            if args.db_collection:
                db_config["collection"] = args.db_collection
        else:
            if args.db_query:
                db_config["query"] = args.db_query
        
        if streamer.add_database_source(args.database, **db_config):
            sources_added += 1
            print(f"✅ Added database source: {args.database}")
    
    # File source
    if args.file:
        if streamer.add_file_source(args.file, args.file_interval):
            sources_added += 1
            print(f"✅ Added file source: {args.file}")
    
    # Webhook source
    if args.webhook:
        if streamer.add_webhook_source(args.webhook_port, args.webhook_path):
            sources_added += 1
            print(f"✅ Added webhook source: port {args.webhook_port}")
    
    # API source
    if args.api_url:
        headers = {}
        params = {}
        
        if args.api_headers:
            try:
                headers = json.loads(args.api_headers)
            except:
                print("⚠️  Invalid API headers JSON, ignoring")
        
        if args.api_params:
            try:
                params = json.loads(args.api_params)
            except:
                print("⚠️  Invalid API params JSON, ignoring")
        
        if streamer.add_api_source(args.api_url, args.api_method, headers, params):
            sources_added += 1
            print(f"✅ Added API source: {args.api_url}")
    
    if sources_added == 0:
        print("❌ No data sources configured!")
        print("Use --help to see available options")
        return
    
    # Start streaming
    try:
        asyncio.run(streamer.start_streaming())
    except KeyboardInterrupt:
        print("\n🛑 Stopping data streaming...")

if __name__ == "__main__":
    main()

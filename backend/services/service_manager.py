"""
Service Manager
Unified management script for all Modal services.
"""

import argparse
import subprocess
import sys
import time
from typing import List, Dict, Any

# Service configurations
SERVICES = {
    "lyrics": {
        "module": "services.lyrics_service_unified",
        "description": "Lyrics, prompt, and category generation service",
        "dependencies": []
    },
    "music": {
        "module": "services.music_service_unified", 
        "description": "Music generation service using ACE-Step",
        "dependencies": []
    },
    "image": {
        "module": "services.cover_image_service_unified",
        "description": "Cover image generation service using SDXL-Turbo", 
        "dependencies": []
    },
    "integrated": {
        "module": "services.integrated_service_unified",
        "description": "Integrated orchestration service",
        "dependencies": ["lyrics", "music", "image"]
    }
}

# Legacy service mappings
LEGACY_SERVICES = {
    "lyrics_legacy": "services.lyrics_service",
    "music_legacy": "services.music_service", 
    "image_legacy": "services.cover_image_service",
    "integrated_legacy": "services.integrated_service"
}


def run_service(service_name: str, legacy: bool = False) -> int:
    """Run a specific service"""
    if legacy:
        if service_name not in LEGACY_SERVICES:
            print(f"❌ Legacy service '{service_name}' not found")
            return 1
        module = LEGACY_SERVICES[service_name]
        print(f"🚀 Running legacy service: {service_name}")
    else:
        if service_name not in SERVICES:
            print(f"❌ Service '{service_name}' not found")
            print(f"Available services: {', '.join(SERVICES.keys())}")
            return 1
        
        service_config = SERVICES[service_name]
        module = service_config["module"]
        print(f"🚀 Running unified service: {service_name}")
        print(f"   Description: {service_config['description']}")
        
        if service_config["dependencies"]:
            print(f"   Dependencies: {', '.join(service_config['dependencies'])}")
    
    try:
        cmd = ["modal", "run", "-m", module]
        print(f"   Command: {' '.join(cmd)}")
        print("-" * 60)
        
        result = subprocess.run(cmd, cwd="backend")
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  Service interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Failed to run service: {e}")
        return 1


def deploy_service(service_name: str, legacy: bool = False) -> int:
    """Deploy a specific service"""
    if legacy:
        if service_name not in LEGACY_SERVICES:
            print(f"❌ Legacy service '{service_name}' not found")
            return 1
        module = LEGACY_SERVICES[service_name]
        print(f"🚀 Deploying legacy service: {service_name}")
    else:
        if service_name not in SERVICES:
            print(f"❌ Service '{service_name}' not found")
            return 1
        
        service_config = SERVICES[service_name]
        module = service_config["module"]
        print(f"🚀 Deploying unified service: {service_name}")
        print(f"   Description: {service_config['description']}")
    
    try:
        cmd = ["modal", "deploy", module]
        print(f"   Command: {' '.join(cmd)}")
        print("-" * 60)
        
        result = subprocess.run(cmd, cwd="backend")
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n⚠️  Deployment interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Failed to deploy service: {e}")
        return 1


def list_services():
    """List all available services"""
    print("📋 Available Unified Services:")
    print("=" * 60)
    
    for name, config in SERVICES.items():
        deps = f" (depends on: {', '.join(config['dependencies'])})" if config['dependencies'] else ""
        print(f"  🔧 {name:<12} - {config['description']}{deps}")
    
    print(f"\n📋 Available Legacy Services:")
    print("=" * 60)
    
    for name, module in LEGACY_SERVICES.items():
        print(f"  🔧 {name:<12} - {module}")
    
    print(f"\n💡 Usage:")
    print(f"  python service_manager.py run <service_name>")
    print(f"  python service_manager.py deploy <service_name>")
    print(f"  python service_manager.py run <service_name> --legacy")


def run_all_services(legacy: bool = False):
    """Run all services in dependency order"""
    if legacy:
        print("🚀 Running all legacy services...")
        services_to_run = list(LEGACY_SERVICES.keys())
    else:
        print("🚀 Running all unified services in dependency order...")
        # Sort services by dependencies
        services_to_run = []
        remaining = set(SERVICES.keys())
        
        while remaining:
            # Find services with no unmet dependencies
            ready = []
            for service in remaining:
                deps = SERVICES[service]["dependencies"]
                if all(dep in services_to_run for dep in deps):
                    ready.append(service)
            
            if not ready:
                print("❌ Circular dependency detected!")
                return 1
            
            services_to_run.extend(ready)
            remaining -= set(ready)
    
    print(f"📋 Service execution order: {' → '.join(services_to_run)}")
    print("-" * 60)
    
    for service in services_to_run:
        print(f"\n🔄 Starting {service}...")
        result = run_service(service, legacy)
        if result != 0:
            print(f"❌ Service {service} failed with exit code {result}")
            return result
        time.sleep(2)  # Brief pause between services
    
    print("\n✅ All services completed!")
    return 0


def deploy_all_services(legacy: bool = False):
    """Deploy all services"""
    if legacy:
        print("🚀 Deploying all legacy services...")
        services_to_deploy = list(LEGACY_SERVICES.keys())
    else:
        print("🚀 Deploying all unified services...")
        services_to_deploy = list(SERVICES.keys())
    
    print(f"📋 Services to deploy: {', '.join(services_to_deploy)}")
    print("-" * 60)
    
    for service in services_to_deploy:
        print(f"\n🔄 Deploying {service}...")
        result = deploy_service(service, legacy)
        if result != 0:
            print(f"❌ Deployment of {service} failed with exit code {result}")
            return result
        time.sleep(1)  # Brief pause between deployments
    
    print("\n✅ All services deployed!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Unified service manager for Modal services")
    parser.add_argument("action", choices=["run", "deploy", "list", "run-all", "deploy-all"],
                       help="Action to perform")
    parser.add_argument("service", nargs="?", help="Service name (required for run/deploy)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy service versions")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_services()
        return 0
    
    elif args.action == "run-all":
        return run_all_services(args.legacy)
    
    elif args.action == "deploy-all":
        return deploy_all_services(args.legacy)
    
    elif args.action in ["run", "deploy"]:
        if not args.service:
            print(f"❌ Service name required for {args.action} action")
            parser.print_help()
            return 1
        
        if args.action == "run":
            return run_service(args.service, args.legacy)
        else:
            return deploy_service(args.service, args.legacy)
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
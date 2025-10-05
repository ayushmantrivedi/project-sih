"""
Comprehensive monitoring and analytics system for HealthBot AI Chatbot
"""
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from database.models import get_db_manager
from utils import get_logger
import psutil
import os

logger = get_logger("analytics")

class HealthBotAnalytics:
    """Analytics and monitoring system for HealthBot"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.metrics = defaultdict(int)
        self.start_time = time.time()
        
    def track_message(self, platform, user_id, message_type, response_type=None, confidence=None):
        """Track a message interaction"""
        try:
            # Update metrics
            self.metrics[f"{platform}_messages"] += 1
            self.metrics[f"{platform}_{message_type}_messages"] += 1
            
            if response_type:
                self.metrics[f"{platform}_{response_type}_responses"] += 1
            
            if confidence:
                self.metrics[f"{platform}_avg_confidence"] = (
                    (self.metrics[f"{platform}_avg_confidence"] * 
                     (self.metrics[f"{platform}_messages"] - 1) + confidence) / 
                    self.metrics[f"{platform}_messages"]
                )
            
            # Log to database
            self._log_to_database(platform, user_id, message_type, response_type, confidence)
            
        except Exception as e:
            logger.error(f"Error tracking message: {str(e)}")
    
    def track_api_call(self, api_name, success=True, response_time=None):
        """Track API call"""
        try:
            self.metrics[f"api_{api_name}_calls"] += 1
            
            if success:
                self.metrics[f"api_{api_name}_success"] += 1
            else:
                self.metrics[f"api_{api_name}_errors"] += 1
            
            if response_time:
                self.metrics[f"api_{api_name}_avg_response_time"] = (
                    (self.metrics[f"api_{api_name}_avg_response_time"] * 
                     (self.metrics[f"api_{api_name}_calls"] - 1) + response_time) / 
                    self.metrics[f"api_{api_name}_calls"]
                )
            
        except Exception as e:
            logger.error(f"Error tracking API call: {str(e)}")
    
    def track_error(self, error_type, platform=None, user_id=None):
        """Track errors"""
        try:
            self.metrics[f"errors_{error_type}"] += 1
            
            if platform:
                self.metrics[f"errors_{platform}_{error_type}"] += 1
            
            logger.error(f"Error tracked: {error_type} on {platform} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error tracking error: {str(e)}")
    
    def get_system_metrics(self):
        """Get system performance metrics"""
        try:
            # CPU and Memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process info
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / 1024 / 1024,
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / 1024 / 1024 / 1024
                },
                "process": {
                    "memory_mb": process_memory,
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads()
                },
                "uptime_seconds": time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    def get_usage_metrics(self, hours=24):
        """Get usage metrics for the last N hours"""
        try:
            if not self.db_manager:
                return {}
            
            # Get analytics from database
            analytics = self.db_manager.get_analytics(days=hours/24)
            
            # Aggregate metrics
            total_messages = sum(a.total_messages for a in analytics)
            total_users = sum(a.unique_users for a in analytics)
            total_predictions = sum(a.predictions_made for a in analytics)
            total_api_calls = sum(a.api_calls for a in analytics)
            total_errors = sum(a.error_count for a in analytics)
            
            # Platform breakdown
            platform_metrics = defaultdict(lambda: {
                'messages': 0, 'users': 0, 'predictions': 0, 'api_calls': 0, 'errors': 0
            })
            
            for a in analytics:
                platform_metrics[a.platform]['messages'] += a.total_messages
                platform_metrics[a.platform]['users'] += a.unique_users
                platform_metrics[a.platform]['predictions'] += a.predictions_made
                platform_metrics[a.platform]['api_calls'] += a.api_calls
                platform_metrics[a.platform]['errors'] += a.error_count
            
            return {
                "summary": {
                    "total_messages": total_messages,
                    "total_users": total_users,
                    "total_predictions": total_predictions,
                    "total_api_calls": total_api_calls,
                    "total_errors": total_errors,
                    "error_rate": total_errors / max(total_messages, 1) * 100
                },
                "platforms": dict(platform_metrics),
                "time_period_hours": hours
            }
            
        except Exception as e:
            logger.error(f"Error getting usage metrics: {str(e)}")
            return {}
    
    def get_health_status(self):
        """Get overall health status"""
        try:
            system_metrics = self.get_system_metrics()
            usage_metrics = self.get_usage_metrics(1)  # Last hour
            
            # Health checks
            health_checks = {
                "database": self._check_database_health(),
                "memory": system_metrics.get("system", {}).get("memory_percent", 0) < 90,
                "cpu": system_metrics.get("system", {}).get("cpu_percent", 0) < 90,
                "disk": system_metrics.get("system", {}).get("disk_percent", 0) < 90,
                "error_rate": usage_metrics.get("summary", {}).get("error_rate", 0) < 5
            }
            
            # Overall health
            overall_health = all(health_checks.values())
            
            return {
                "status": "healthy" if overall_health else "unhealthy",
                "checks": health_checks,
                "system_metrics": system_metrics,
                "usage_metrics": usage_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _check_database_health(self):
        """Check database connectivity"""
        try:
            if not self.db_manager:
                return False
            
            session = self.db_manager.get_session()
            session.execute("SELECT 1")
            session.close()
            return True
            
        except Exception:
            return False
    
    def _log_to_database(self, platform, user_id, message_type, response_type, confidence):
        """Log metrics to database"""
        try:
            if not self.db_manager:
                return
            
            # This would be implemented to log to analytics table
            # For now, just log to console
            logger.info(f"Analytics: {platform} - {message_type} - {response_type}")
            
        except Exception as e:
            logger.error(f"Error logging to database: {str(e)}")
    
    def generate_report(self, hours=24):
        """Generate comprehensive analytics report"""
        try:
            system_metrics = self.get_system_metrics()
            usage_metrics = self.get_usage_metrics(hours)
            health_status = self.get_health_status()
            
            report = {
                "report_generated_at": datetime.utcnow().isoformat(),
                "time_period_hours": hours,
                "health_status": health_status,
                "system_metrics": system_metrics,
                "usage_metrics": usage_metrics,
                "recommendations": self._generate_recommendations(health_status, usage_metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, health_status, usage_metrics):
        """Generate recommendations based on metrics"""
        recommendations = []
        
        # Check system health
        if not health_status.get("checks", {}).get("memory", True):
            recommendations.append("High memory usage detected. Consider scaling up or optimizing memory usage.")
        
        if not health_status.get("checks", {}).get("cpu", True):
            recommendations.append("High CPU usage detected. Consider scaling up or optimizing CPU usage.")
        
        if not health_status.get("checks", {}).get("disk", True):
            recommendations.append("High disk usage detected. Consider cleaning up logs or scaling storage.")
        
        # Check error rate
        error_rate = usage_metrics.get("summary", {}).get("error_rate", 0)
        if error_rate > 5:
            recommendations.append(f"High error rate detected ({error_rate:.1f}%). Check logs for issues.")
        
        # Check database health
        if not health_status.get("checks", {}).get("database", True):
            recommendations.append("Database connectivity issues detected. Check database configuration.")
        
        return recommendations

# Global analytics instance
analytics = HealthBotAnalytics()

def get_analytics():
    """Get analytics instance"""
    return analytics
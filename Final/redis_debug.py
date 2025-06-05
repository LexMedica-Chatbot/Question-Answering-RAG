#!/usr/bin/env python3
"""
Redis Debug Script - Test dan inspect Redis cache content
Untuk debugging dan monitoring cache performance
"""

import redis
import json
import time
import os
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RedisDebugger:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis debugger"""
        self.redis_url = redis_url
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            self.connected = True
            print(f"✅ Connected to Redis: {redis_url}")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.connected = False
            self.redis_client = None

    def get_all_keys(self) -> Dict[str, List[str]]:
        """Get all Redis keys by category"""
        if not self.connected:
            return {}

        try:
            all_keys = [key.decode() for key in self.redis_client.keys("*")]

            categorized = {
                "exact": [k for k in all_keys if k.startswith("exact:")],
                "semantic": [k for k in all_keys if k.startswith("semantic:")],
                "embedding": [k for k in all_keys if k.startswith("embedding:")],
                "docs": [k for k in all_keys if k.startswith("docs:")],
                "other": [
                    k
                    for k in all_keys
                    if not any(
                        k.startswith(prefix)
                        for prefix in ["exact:", "semantic:", "embedding:", "docs:"]
                    )
                ],
            }

            return categorized
        except Exception as e:
            print(f"Error getting keys: {e}")
            return {}

    def inspect_cache_key(self, key: str) -> Dict[str, Any]:
        """Inspect a specific cache key"""
        if not self.connected:
            return {}

        try:
            # Get TTL
            ttl = self.redis_client.ttl(key)

            # Get value
            value = self.redis_client.get(key)
            if not value:
                return {"error": "Key not found or expired"}

            # Try to parse as JSON
            try:
                parsed_value = json.loads(value)
                value_type = "json"
                value_size = len(value)
            except json.JSONDecodeError:
                parsed_value = (
                    value.decode() if isinstance(value, bytes) else str(value)
                )
                value_type = "string"
                value_size = len(parsed_value)

            return {
                "key": key,
                "ttl_seconds": ttl,
                "value_type": value_type,
                "value_size_bytes": value_size,
                "value_preview": (
                    str(parsed_value)[:200] + "..."
                    if len(str(parsed_value)) > 200
                    else parsed_value
                ),
                "full_value": parsed_value,
            }
        except Exception as e:
            return {"error": f"Error inspecting key: {e}"}

    def test_semantic_similarity(self) -> List[Dict[str, Any]]:
        """Test semantic similarity between cached embeddings"""
        if not self.connected:
            return []

        try:
            # Get all embedding keys
            embedding_keys = [k for k in self.redis_client.keys("embedding:*")]
            embeddings_data = []

            for key in embedding_keys[:10]:  # Limit untuk performance
                try:
                    embedding_data = self.redis_client.get(key)
                    if embedding_data:
                        embedding = json.loads(embedding_data)
                        if isinstance(embedding, list) and len(embedding) > 0:
                            embeddings_data.append(
                                {
                                    "key": key.decode(),
                                    "embedding": embedding,
                                    "dimensions": len(embedding),
                                }
                            )
                except Exception as e:
                    print(f"Error processing embedding {key}: {e}")

            # Calculate similarities
            similarities = []
            for i, emb1 in enumerate(embeddings_data):
                for j, emb2 in enumerate(embeddings_data):
                    if i < j:  # Avoid duplicates
                        try:
                            vec1 = np.array(emb1["embedding"]).reshape(1, -1)
                            vec2 = np.array(emb2["embedding"]).reshape(1, -1)
                            similarity = cosine_similarity(vec1, vec2)[0][0]

                            similarities.append(
                                {
                                    "key1": emb1["key"],
                                    "key2": emb2["key"],
                                    "similarity": float(similarity),
                                    "high_similarity": similarity > 0.85,
                                }
                            )
                        except Exception as e:
                            print(f"Error calculating similarity: {e}")

            # Sort by similarity descending
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:20]  # Top 20

        except Exception as e:
            print(f"Error testing semantic similarity: {e}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self.connected:
            return {"connected": False}

        try:
            # Basic Redis info
            info = self.redis_client.info()

            # Key counts by category
            keys_by_category = self.get_all_keys()

            # Memory usage breakdown
            total_memory = 0
            memory_by_category = {}

            for category, keys in keys_by_category.items():
                category_memory = 0
                for key in keys:
                    try:
                        memory_usage = self.redis_client.memory_usage(key)
                        if memory_usage:
                            category_memory += memory_usage
                    except:
                        pass
                memory_by_category[category] = category_memory
                total_memory += category_memory

            return {
                "connected": True,
                "redis_info": {
                    "version": info.get("redis_version", "unknown"),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                },
                "cache_statistics": {
                    "total_keys": sum(len(keys) for keys in keys_by_category.values()),
                    "keys_by_category": {
                        cat: len(keys) for cat, keys in keys_by_category.items()
                    },
                    "memory_by_category_bytes": memory_by_category,
                    "total_cache_memory_bytes": total_memory,
                },
                "hit_rate": self._calculate_hit_rate(info),
            }
        except Exception as e:
            return {"connected": True, "error": f"Error getting stats: {e}"}

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0

    def monitor_cache_activity(self, duration_seconds: int = 30):
        """Monitor Redis activity in real-time"""
        if not self.connected:
            print("❌ Not connected to Redis")
            return

        print(f"🔍 Monitoring Redis activity for {duration_seconds} seconds...")
        print("Press Ctrl+C to stop early")

        # Get initial stats
        initial_stats = self.get_cache_stats()
        initial_keys = initial_stats["cache_statistics"]["total_keys"]

        try:
            time.sleep(duration_seconds)
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")

        # Get final stats
        final_stats = self.get_cache_stats()
        final_keys = final_stats["cache_statistics"]["total_keys"]

        print(f"\n📊 Monitoring Results:")
        print(f"   Keys before: {initial_keys}")
        print(f"   Keys after: {final_keys}")
        print(f"   Keys change: {final_keys - initial_keys:+d}")
        print(f"   Hit rate: {final_stats['hit_rate']:.2f}%")

    def clear_cache_category(self, category: str) -> bool:
        """Clear specific category of cache"""
        if not self.connected:
            return False

        try:
            keys = self.redis_client.keys(f"{category}:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                print(f"✅ Deleted {deleted} keys from category '{category}'")
                return True
            else:
                print(f"ℹ️ No keys found in category '{category}'")
                return True
        except Exception as e:
            print(f"❌ Error clearing category '{category}': {e}")
            return False

    def generate_report(self) -> str:
        """Generate comprehensive Redis cache report"""
        if not self.connected:
            return "❌ Not connected to Redis"

        report = ["🔍 REDIS CACHE ANALYSIS REPORT", "=" * 50]

        # Basic stats
        stats = self.get_cache_stats()
        redis_info = stats.get("redis_info", {})
        cache_stats = stats.get("cache_statistics", {})

        report.append(f"\n📊 Redis Instance Info:")
        report.append(f"   Version: {redis_info.get('version', 'unknown')}")
        report.append(f"   Uptime: {redis_info.get('uptime_seconds', 0)} seconds")
        report.append(f"   Memory Usage: {redis_info.get('used_memory_human', '0B')}")
        report.append(
            f"   Peak Memory: {redis_info.get('used_memory_peak_human', '0B')}"
        )
        report.append(f"   Hit Rate: {stats.get('hit_rate', 0):.2f}%")

        report.append(f"\n🗂️ Cache Categories:")
        keys_by_cat = cache_stats.get("keys_by_category", {})
        for category, count in keys_by_cat.items():
            if count > 0:
                report.append(f"   {category}: {count} keys")

        # Sample some keys from each category
        keys_by_category = self.get_all_keys()
        for category, keys in keys_by_category.items():
            if keys:
                report.append(f"\n🔑 {category.title()} Cache Samples:")
                for key in keys[:3]:  # Show first 3 keys
                    key_info = self.inspect_cache_key(key)
                    ttl = key_info.get("ttl_seconds", -1)
                    size = key_info.get("value_size_bytes", 0)
                    ttl_str = (
                        f"{ttl}s"
                        if ttl > 0
                        else "no expiry" if ttl == -1 else "expired"
                    )
                    report.append(f"   • {key} (TTL: {ttl_str}, Size: {size}B)")

        # Semantic similarity analysis
        similarities = self.test_semantic_similarity()
        if similarities:
            report.append(f"\n🧠 Semantic Similarity Analysis:")
            high_sim = [s for s in similarities if s["high_similarity"]]
            report.append(f"   High similarity pairs (>0.85): {len(high_sim)}")
            if high_sim:
                report.append(f"   Highest similarity: {high_sim[0]['similarity']:.3f}")

        return "\n".join(report)


def main():
    """Main function untuk Redis debugging"""
    print("🔧 Redis Cache Debugger")
    print("=" * 50)

    # Initialize debugger
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    debugger = RedisDebugger(redis_url)

    if not debugger.connected:
        print("❌ Cannot connect to Redis. Make sure Redis is running.")
        return

    while True:
        print("\n🛠️ Redis Debug Menu:")
        print("1. 📊 Show cache statistics")
        print("2. 🗂️ List all keys by category")
        print("3. 🔍 Inspect specific key")
        print("4. 🧠 Test semantic similarity")
        print("5. 📋 Generate full report")
        print("6. 🔍 Monitor cache activity")
        print("7. 🗑️ Clear cache category")
        print("8. 🚪 Exit")

        choice = input("\nSelect option (1-8): ").strip()

        if choice == "1":
            print("\n📊 Cache Statistics:")
            stats = debugger.get_cache_stats()
            print(json.dumps(stats, indent=2))

        elif choice == "2":
            print("\n🗂️ Keys by Category:")
            keys = debugger.get_all_keys()
            for category, key_list in keys.items():
                print(f"   {category}: {len(key_list)} keys")
                if key_list:
                    for key in key_list[:5]:  # Show first 5
                        print(f"     • {key}")
                    if len(key_list) > 5:
                        print(f"     ... and {len(key_list) - 5} more")

        elif choice == "3":
            key = input("Enter key to inspect: ").strip()
            if key:
                print(f"\n🔍 Key: {key}")
                info = debugger.inspect_cache_key(key)
                print(json.dumps(info, indent=2))

        elif choice == "4":
            print("\n🧠 Testing Semantic Similarity...")
            similarities = debugger.test_semantic_similarity()
            if similarities:
                print(f"Found {len(similarities)} similarity pairs:")
                for sim in similarities[:10]:
                    print(f"   {sim['similarity']:.3f}: {sim['key1']} ↔ {sim['key2']}")
            else:
                print("No embeddings found or error occurred")

        elif choice == "5":
            print("\n📋 Generating Full Report...")
            report = debugger.generate_report()
            print(report)

            # Save to file
            with open("redis_cache_report.txt", "w") as f:
                f.write(report)
            print(f"\n💾 Report saved to: redis_cache_report.txt")

        elif choice == "6":
            duration = input("Monitor duration in seconds (default 30): ").strip()
            duration = int(duration) if duration.isdigit() else 30
            debugger.monitor_cache_activity(duration)

        elif choice == "7":
            print("\nAvailable categories: exact, semantic, embedding, docs")
            category = input("Enter category to clear: ").strip()
            if category:
                confirm = (
                    input(f"Are you sure you want to clear '{category}' cache? (y/N): ")
                    .strip()
                    .lower()
                )
                if confirm == "y":
                    debugger.clear_cache_category(category)

        elif choice == "8":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid option. Please try again.")


if __name__ == "__main__":
    main()

#!/bin/bash

echo "🧪 Testing DSPy Agent Frontend API Endpoints"
echo "=============================================="

# Test all API endpoints
echo "📊 Testing API endpoints..."

echo "1. Status endpoint:"
curl -s http://localhost:3000/api/status | jq '.status' 2>/dev/null || echo "❌ Failed"

echo "2. Metrics endpoint:"
curl -s http://localhost:3000/api/metrics | jq '.cpu_usage' 2>/dev/null || echo "❌ Failed"

echo "3. RL Metrics endpoint (with avg_reward):"
curl -s http://localhost:3000/api/rl-metrics | jq '.avg_reward' 2>/dev/null || echo "❌ Failed"

echo "4. Bus Metrics endpoint:"
curl -s http://localhost:3000/api/bus-metrics | jq '.total_messages' 2>/dev/null || echo "❌ Failed"

echo "5. Logs endpoint:"
curl -s http://localhost:3000/api/logs | jq '.total_logs' 2>/dev/null || echo "❌ Failed"

echo "6. Kafka Topics endpoint:"
curl -s http://localhost:3000/api/kafka-topics | jq '.total_topics' 2>/dev/null || echo "❌ Failed"

echo ""
echo "✅ All endpoints tested!"
echo ""
echo "🌐 Frontend URL: http://localhost:3000"
echo ""
echo "💡 If you're still seeing errors in the browser:"
echo "   1. Hard refresh the page (Ctrl+F5 or Cmd+Shift+R)"
echo "   2. Clear browser cache"
echo "   3. Open Developer Tools and check the Network tab"
echo "   4. Look for any failed API requests"

import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

const QuantumAnalyticsDashboard = ({ quantumData, hardwareData }) => {
  const [activeMetric, setActiveMetric] = useState('fidelity');
  const [timeRange, setTimeRange] = useState('1h');
  const [realTimeData, setRealTimeData] = useState([]);
  const svgRefs = {
    fidelity: useRef(),
    entanglement: useRef(),
    coherence: useRef(),
    gateErrors: useRef()
  };

  // Quantum metrics available
  const quantumMetrics = {
    fidelity: {
      name: 'Quantum Fidelity',
      icon: '🎯',
      unit: '%',
      description: 'Measure of quantum state accuracy',
      color: '#3b82f6'
    },
    entanglement: {
      name: 'Entanglement Measure',
      icon: '🔗',
      unit: 'bits',
      description: 'Quantum entanglement entropy',
      color: '#10b981'
    },
    coherence: {
      name: 'Coherence Time',
      icon: '⏱️',
      unit: 'μs',
      description: 'Quantum coherence duration',
      color: '#f59e0b'
    },
    gateErrors: {
      name: 'Gate Error Rates',
      icon: '⚠️',
      unit: '%',
      description: 'Quantum gate error probability',
      color: '#ef4444'
    }
  };

  useEffect(() => {
    // Simulate real-time data updates
    const interval = setInterval(() => {
      generateRealTimeData();
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (realTimeData.length > 0) {
      renderQuantumMetrics();
    }
  }, [realTimeData, activeMetric, timeRange]);

  const generateRealTimeData = () => {
    const now = new Date();
    const newDataPoint = {
      timestamp: now,
      fidelity: 85 + Math.random() * 10 + Math.sin(now.getTime() / 10000) * 5,
      entanglement: 1.2 + Math.random() * 0.8,
      coherence: 50 + Math.random() * 20,
      gateErrors: 0.1 + Math.random() * 0.05,
      quantumVolume: Math.floor(32 + Math.random() * 32),
      circuitDepth: Math.floor(10 + Math.random() * 20),
      shots: 1000 + Math.floor(Math.random() * 9000)
    };

    setRealTimeData(prev => {
      const updated = [...prev, newDataPoint];
      // Keep only last 100 data points
      return updated.slice(-100);
    });
  };

  const renderQuantumMetrics = () => {
    Object.keys(svgRefs).forEach(metric => {
      renderMetricChart(metric);
    });
  };

  const renderMetricChart = (metric) => {
    const svg = d3.select(svgRefs[metric].current);
    svg.selectAll("*").remove();

    if (realTimeData.length === 0) return;

    const margin = { top: 20, right: 30, bottom: 30, left: 40 };
    const width = 300 - margin.left - margin.right;
    const height = 150 - margin.top - margin.bottom;

    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Filter data based on time range
    const now = new Date();
    const timeRangeMs = {
      '1h': 60 * 60 * 1000,
      '6h': 6 * 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000
    };

    const filteredData = realTimeData.filter(d => 
      now - d.timestamp <= timeRangeMs[timeRange]
    );

    if (filteredData.length === 0) return;

    const xScale = d3.scaleTime()
      .domain(d3.extent(filteredData, d => d.timestamp))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(filteredData, d => d[metric]))
      .nice()
      .range([height, 0]);

    // Add gradient
    const gradient = g.append("defs")
      .append("linearGradient")
      .attr("id", `gradient-${metric}`)
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("x1", 0).attr("y1", height)
      .attr("x2", 0).attr("y2", 0);

    gradient.append("stop")
      .attr("offset", "0%")
      .attr("stop-color", quantumMetrics[metric].color)
      .attr("stop-opacity", 0.1);

    gradient.append("stop")
      .attr("offset", "100%")
      .attr("stop-color", quantumMetrics[metric].color)
      .attr("stop-opacity", 0.8);

    // Create line and area
    const line = d3.line()
      .x(d => xScale(d.timestamp))
      .y(d => yScale(d[metric]))
      .curve(d3.curveMonotoneX);

    const area = d3.area()
      .x(d => xScale(d.timestamp))
      .y0(height)
      .y1(d => yScale(d[metric]))
      .curve(d3.curveMonotoneX);

    // Add area
    g.append("path")
      .datum(filteredData)
      .attr("fill", `url(#gradient-${metric})`)
      .attr("d", area);

    // Add line
    g.append("path")
      .datum(filteredData)
      .attr("fill", "none")
      .attr("stroke", quantumMetrics[metric].color)
      .attr("stroke-width", 2)
      .attr("d", line);

    // Add dots for recent points
    g.selectAll(".dot")
      .data(filteredData.slice(-10))
      .enter().append("circle")
      .attr("class", "dot")
      .attr("cx", d => xScale(d.timestamp))
      .attr("cy", d => yScale(d[metric]))
      .attr("r", 3)
      .attr("fill", quantumMetrics[metric].color);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.timeFormat("%H:%M")));

    g.append("g")
      .call(d3.axisLeft(yScale).ticks(5));

    // Add current value indicator
    if (filteredData.length > 0) {
      const currentValue = filteredData[filteredData.length - 1][metric];
      g.append("text")
        .attr("x", width - 5)
        .attr("y", yScale(currentValue) - 5)
        .attr("text-anchor", "end")
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .attr("fill", quantumMetrics[metric].color)
        .text(`${currentValue.toFixed(2)}${quantumMetrics[metric].unit}`);
    }
  };

  const renderQuantumCircuitComplexity = () => {
    // This would render a more complex visualization of quantum circuit complexity
    return (
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg">
          <h4 className="font-semibold text-indigo-800 mb-2">Circuit Depth</h4>
          <div className="text-2xl font-bold text-indigo-600">
            {realTimeData.length > 0 ? realTimeData[realTimeData.length - 1].circuitDepth : '--'}
          </div>
          <div className="text-sm text-indigo-600">layers</div>
        </div>
        
        <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg">
          <h4 className="font-semibold text-emerald-800 mb-2">Quantum Volume</h4>
          <div className="text-2xl font-bold text-emerald-600">
            {realTimeData.length > 0 ? realTimeData[realTimeData.length - 1].quantumVolume : '--'}
          </div>
          <div className="text-sm text-emerald-600">qubits</div>
        </div>
      </div>
    );
  };

  const renderQuantumAdvantageMetrics = () => {
    if (realTimeData.length === 0) return null;

    const latest = realTimeData[realTimeData.length - 1];
    const quantumAdvantage = (latest.fidelity / 100) * (latest.entanglement / 2) * (100 - latest.gateErrors) / 100;

    return (
      <div className="space-y-4">
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-6 rounded-lg">
          <h4 className="font-semibold text-purple-800 mb-3">Quantum Advantage Score</h4>
          <div className="flex items-center gap-4">
            <div className="text-4xl font-bold text-purple-600">
              {(quantumAdvantage * 100).toFixed(1)}
            </div>
            <div className="flex-1">
              <div className="w-full bg-purple-200 rounded-full h-3">
                <div 
                  className="bg-purple-600 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${quantumAdvantage * 100}%` }}
                ></div>
              </div>
              <div className="text-sm text-purple-600 mt-1">
                Composite quantum performance metric
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-lg font-bold text-blue-600">{latest.fidelity.toFixed(1)}%</div>
            <div className="text-sm text-blue-600">Fidelity</div>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-lg font-bold text-green-600">{latest.entanglement.toFixed(2)}</div>
            <div className="text-sm text-green-600">Entanglement</div>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-lg font-bold text-red-600">{(latest.gateErrors * 100).toFixed(2)}%</div>
            <div className="text-sm text-red-600">Gate Errors</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">⚛️</span>
            Quantum Analytics Dashboard
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Time Range Selector */}
          <div className="flex items-center gap-4 mb-6">
            <span className="font-medium">Time Range:</span>
            {['1h', '6h', '24h', '7d'].map(range => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 rounded-md text-sm ${
                  timeRange === range
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {range}
              </button>
            ))}
            <div className="ml-auto flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600">Live Data</span>
            </div>
          </div>

          {/* Real-time Quantum Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {Object.entries(quantumMetrics).map(([key, metric]) => (
              <Card key={key} className="cursor-pointer hover:shadow-md transition-shadow">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-2xl">{metric.icon}</span>
                    <Badge variant={activeMetric === key ? "primary" : "secondary"}>
                      Live
                    </Badge>
                  </div>
                  <h4 className="font-medium text-sm mb-1">{metric.name}</h4>
                  <div className="text-xs text-gray-600 mb-3">{metric.description}</div>
                  <svg ref={svgRefs[key]}></svg>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Quantum Advantage Metrics */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-3">Quantum Advantage Analysis</h3>
            {renderQuantumAdvantageMetrics()}
          </div>

          {/* Circuit Complexity */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-3">Circuit Complexity</h3>
            {renderQuantumCircuitComplexity()}
          </div>

          {/* Hardware Status */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-r from-cyan-50 to-blue-50 p-4 rounded-lg">
              <h4 className="font-semibold text-cyan-800 mb-2">Hardware Status</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">IBM Quantum</span>
                  <Badge variant="success">Online</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">IonQ</span>
                  <Badge variant="success">Online</Badge>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Rigetti</span>
                  <Badge variant="warning">Maintenance</Badge>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-orange-50 to-red-50 p-4 rounded-lg">
              <h4 className="font-semibold text-orange-800 mb-2">Queue Status</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Current Jobs</span>
                  <span className="font-medium">3</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Avg Wait Time</span>
                  <span className="font-medium">2.3 min</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Success Rate</span>
                  <span className="font-medium">94.2%</span>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg">
              <h4 className="font-semibold text-green-800 mb-2">Cost Analysis</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Today's Usage</span>
                  <span className="font-medium">$12.45</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Monthly Budget</span>
                  <span className="font-medium">$500.00</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Remaining</span>
                  <span className="font-medium text-green-600">$487.55</span>
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-center gap-4 mt-6">
            <Button onClick={() => setRealTimeData([])}>
              🔄 Reset Data
            </Button>
            <Button variant="outline">
              📊 Export Analytics
            </Button>
            <Button variant="outline">
              ⚙️ Configure Alerts
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default QuantumAnalyticsDashboard;
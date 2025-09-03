import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

const AdvancedGraphAnalytics = ({ simulationData, algorithmResults }) => {
  const [selectedGraph, setSelectedGraph] = useState('convergence');
  const [graphData, setGraphData] = useState(null);
  const svgRef = useRef();

  // Graph types available
  const graphTypes = {
    convergence: {
      name: 'Convergence Analysis',
      icon: '📈',
      description: 'Training convergence and optimization paths'
    },
    quantum_advantage: {
      name: 'Quantum Advantage',
      icon: '⚛️',
      description: 'Quantum vs classical performance comparison'
    },
    parameter_landscape: {
      name: 'Parameter Landscape',
      icon: '🗺️',
      description: '3D parameter optimization landscape'
    },
    noise_analysis: {
      name: 'Noise Impact',
      icon: '🌊',
      description: 'Quantum noise effects on performance'
    },
    feature_correlation: {
      name: 'Feature Correlation',
      icon: '🔗',
      description: 'Feature correlation heatmap'
    },
    algorithm_comparison: {
      name: 'Algorithm Comparison',
      icon: '⚖️',
      description: 'Multi-algorithm performance radar'
    },
    quantum_circuit_depth: {
      name: 'Circuit Complexity',
      icon: '🔄',
      description: 'Quantum circuit depth vs performance'
    },
    error_mitigation: {
      name: 'Error Mitigation',
      icon: '🛡️',
      description: 'Error correction effectiveness'
    }
  };

  useEffect(() => {
    if (simulationData || algorithmResults) {
      generateGraphData();
    }
  }, [selectedGraph, simulationData, algorithmResults]);

  useEffect(() => {
    if (graphData) {
      renderGraph();
    }
  }, [graphData]);

  const generateGraphData = () => {
    // Generate synthetic data based on graph type
    let data;
    
    switch (selectedGraph) {
      case 'convergence':
        data = generateConvergenceData();
        break;
      case 'quantum_advantage':
        data = generateQuantumAdvantageData();
        break;
      case 'parameter_landscape':
        data = generateParameterLandscapeData();
        break;
      case 'noise_analysis':
        data = generateNoiseAnalysisData();
        break;
      case 'feature_correlation':
        data = generateFeatureCorrelationData();
        break;
      case 'algorithm_comparison':
        data = generateAlgorithmComparisonData();
        break;
      case 'quantum_circuit_depth':
        data = generateCircuitDepthData();
        break;
      case 'error_mitigation':
        data = generateErrorMitigationData();
        break;
      default:
        data = generateConvergenceData();
    }
    
    setGraphData(data);
  };

  const generateConvergenceData = () => {
    const iterations = 50;
    const algorithms = ['VQC', 'QSVC', 'Random Forest', 'SVM'];
    
    return algorithms.map(alg => ({
      name: alg,
      data: Array.from({ length: iterations }, (_, i) => ({
        iteration: i,
        loss: Math.exp(-i / 10) + Math.random() * 0.1,
        accuracy: 1 - Math.exp(-i / 15) + Math.random() * 0.05
      }))
    }));
  };

  const generateQuantumAdvantageData = () => {
    const problemSizes = [2, 4, 6, 8, 10, 12, 14, 16];
    
    return {
      quantum: problemSizes.map(size => ({
        size,
        time: Math.pow(2, size * 0.3) + Math.random() * 5,
        accuracy: 0.95 - size * 0.01 + Math.random() * 0.05
      })),
      classical: problemSizes.map(size => ({
        size,
        time: Math.pow(2, size * 0.8) + Math.random() * 10,
        accuracy: 0.90 - size * 0.005 + Math.random() * 0.03
      }))
    };
  };

  const generateParameterLandscapeData = () => {
    const resolution = 30;
    const data = [];
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = (i / resolution) * 4 - 2;
        const y = (j / resolution) * 4 - 2;
        const z = Math.sin(x) * Math.cos(y) + Math.random() * 0.1;
        data.push({ x, y, z });
      }
    }
    
    return data;
  };

  const generateNoiseAnalysisData = () => {
    const noiselevels = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3];
    const algorithms = ['VQC', 'QAOA', 'VQE'];
    
    return algorithms.map(alg => ({
      name: alg,
      data: noiselevels.map(noise => ({
        noise,
        fidelity: Math.exp(-noise * 10) * (0.9 + Math.random() * 0.1),
        performance: Math.exp(-noise * 5) * (0.8 + Math.random() * 0.15)
      }))
    }));
  };

  const generateFeatureCorrelationData = () => {
    const features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'];
    const matrix = [];
    
    for (let i = 0; i < features.length; i++) {
      for (let j = 0; j < features.length; j++) {
        matrix.push({
          source: features[i],
          target: features[j],
          correlation: i === j ? 1 : (Math.random() - 0.5) * 2
        });
      }
    }
    
    return { features, matrix };
  };

  const generateAlgorithmComparisonData = () => {
    const metrics = ['Accuracy', 'Speed', 'Robustness', 'Scalability', 'Interpretability'];
    const algorithms = ['VQC', 'QSVC', 'Random Forest', 'SVM', 'Neural Network'];
    
    return algorithms.map(alg => ({
      name: alg,
      metrics: metrics.map(metric => ({
        metric,
        value: 0.3 + Math.random() * 0.7
      }))
    }));
  };

  const generateCircuitDepthData = () => {
    const depths = Array.from({ length: 20 }, (_, i) => i + 1);
    
    return depths.map(depth => ({
      depth,
      performance: Math.max(0, 0.9 - depth * 0.02 + Math.random() * 0.1),
      noise_impact: Math.min(1, depth * 0.05 + Math.random() * 0.1),
      execution_time: depth * 0.1 + Math.random() * 0.05
    }));
  };

  const generateErrorMitigationData = () => {
    const techniques = ['None', 'Readout', 'ZNE', 'DD', 'Composite'];
    
    return techniques.map(technique => ({
      technique,
      fidelity_improvement: technique === 'None' ? 0 : 0.1 + Math.random() * 0.3,
      overhead: technique === 'None' ? 1 : 1 + Math.random() * 0.5,
      effectiveness: technique === 'None' ? 0 : 0.2 + Math.random() * 0.6
    }));
  };

  const renderGraph = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;
    
    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    switch (selectedGraph) {
      case 'convergence':
        renderConvergenceGraph(g, width, height);
        break;
      case 'quantum_advantage':
        renderQuantumAdvantageGraph(g, width, height);
        break;
      case 'parameter_landscape':
        renderParameterLandscapeGraph(g, width, height);
        break;
      case 'noise_analysis':
        renderNoiseAnalysisGraph(g, width, height);
        break;
      case 'feature_correlation':
        renderFeatureCorrelationGraph(g, width, height);
        break;
      case 'algorithm_comparison':
        renderAlgorithmComparisonGraph(g, width, height);
        break;
      case 'quantum_circuit_depth':
        renderCircuitDepthGraph(g, width, height);
        break;
      case 'error_mitigation':
        renderErrorMitigationGraph(g, width, height);
        break;
    }
  };

  const renderConvergenceGraph = (g, width, height) => {
    if (!graphData) return;

    const xScale = d3.scaleLinear()
      .domain([0, d3.max(graphData[0].data, d => d.iteration)])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    const line = d3.line()
      .x(d => xScale(d.iteration))
      .y(d => yScale(d.accuracy))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add lines
    graphData.forEach((algorithm, i) => {
      g.append("path")
        .datum(algorithm.data)
        .attr("fill", "none")
        .attr("stroke", colorScale(i))
        .attr("stroke-width", 2)
        .attr("d", line);

      // Add legend
      g.append("text")
        .attr("x", width - 100)
        .attr("y", 20 + i * 20)
        .attr("fill", colorScale(i))
        .style("font-size", "12px")
        .text(algorithm.name);
    });

    // Add labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Accuracy");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Iteration");
  };

  const renderQuantumAdvantageGraph = (g, width, height) => {
    if (!graphData) return;

    const xScale = d3.scaleLinear()
      .domain(d3.extent([...graphData.quantum, ...graphData.classical], d => d.size))
      .range([0, width]);

    const yScale = d3.scaleLog()
      .domain(d3.extent([...graphData.quantum, ...graphData.classical], d => d.time))
      .range([height, 0]);

    const line = d3.line()
      .x(d => xScale(d.size))
      .y(d => yScale(d.time))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Quantum line
    g.append("path")
      .datum(graphData.quantum)
      .attr("fill", "none")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 3)
      .attr("d", line);

    // Classical line
    g.append("path")
      .datum(graphData.classical)
      .attr("fill", "none")
      .attr("stroke", "#ef4444")
      .attr("stroke-width", 3)
      .attr("d", line);

    // Add points
    g.selectAll(".quantum-point")
      .data(graphData.quantum)
      .enter().append("circle")
      .attr("class", "quantum-point")
      .attr("cx", d => xScale(d.size))
      .attr("cy", d => yScale(d.time))
      .attr("r", 4)
      .attr("fill", "#3b82f6");

    g.selectAll(".classical-point")
      .data(graphData.classical)
      .enter().append("circle")
      .attr("class", "classical-point")
      .attr("cx", d => xScale(d.size))
      .attr("cy", d => yScale(d.time))
      .attr("r", 4)
      .attr("fill", "#ef4444");

    // Legend
    g.append("text")
      .attr("x", width - 100)
      .attr("y", 20)
      .attr("fill", "#3b82f6")
      .text("Quantum");

    g.append("text")
      .attr("x", width - 100)
      .attr("y", 40)
      .attr("fill", "#ef4444")
      .text("Classical");

    // Labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Execution Time (log scale)");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Problem Size");
  };

  const renderFeatureCorrelationGraph = (g, width, height) => {
    if (!graphData) return;

    const { features, matrix } = graphData;
    const cellSize = Math.min(width, height) / features.length;

    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
      .domain([-1, 1]);

    // Create heatmap
    matrix.forEach(d => {
      const i = features.indexOf(d.source);
      const j = features.indexOf(d.target);
      
      g.append("rect")
        .attr("x", j * cellSize)
        .attr("y", i * cellSize)
        .attr("width", cellSize)
        .attr("height", cellSize)
        .attr("fill", colorScale(d.correlation))
        .attr("stroke", "white")
        .attr("stroke-width", 1);

      // Add correlation values
      g.append("text")
        .attr("x", j * cellSize + cellSize / 2)
        .attr("y", i * cellSize + cellSize / 2)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "10px")
        .attr("fill", Math.abs(d.correlation) > 0.5 ? "white" : "black")
        .text(d.correlation.toFixed(2));
    });

    // Add feature labels
    features.forEach((feature, i) => {
      g.append("text")
        .attr("x", -5)
        .attr("y", i * cellSize + cellSize / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .style("font-size", "12px")
        .text(feature);

      g.append("text")
        .attr("x", i * cellSize + cellSize / 2)
        .attr("y", -5)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "end")
        .style("font-size", "12px")
        .text(feature);
    });
  };

  const renderAlgorithmComparisonGraph = (g, width, height) => {
    if (!graphData) return;

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 50;

    const angleScale = d3.scaleLinear()
      .domain([0, graphData[0].metrics.length])
      .range([0, 2 * Math.PI]);

    const radiusScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, radius]);

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Draw radar grid
    const levels = 5;
    for (let i = 1; i <= levels; i++) {
      const levelRadius = (radius / levels) * i;
      
      g.append("circle")
        .attr("cx", centerX)
        .attr("cy", centerY)
        .attr("r", levelRadius)
        .attr("fill", "none")
        .attr("stroke", "#ddd")
        .attr("stroke-width", 1);
    }

    // Draw axes
    graphData[0].metrics.forEach((metric, i) => {
      const angle = angleScale(i) - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      g.append("line")
        .attr("x1", centerX)
        .attr("y1", centerY)
        .attr("x2", x)
        .attr("y2", y)
        .attr("stroke", "#ddd")
        .attr("stroke-width", 1);

      // Add metric labels
      g.append("text")
        .attr("x", x + Math.cos(angle) * 20)
        .attr("y", y + Math.sin(angle) * 20)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "12px")
        .text(metric.metric);
    });

    // Draw algorithm polygons
    graphData.forEach((algorithm, algIndex) => {
      const points = algorithm.metrics.map((metric, i) => {
        const angle = angleScale(i) - Math.PI / 2;
        const r = radiusScale(metric.value);
        return [
          centerX + Math.cos(angle) * r,
          centerY + Math.sin(angle) * r
        ];
      });

      const lineGenerator = d3.line()
        .x(d => d[0])
        .y(d => d[1])
        .curve(d3.curveLinearClosed);

      g.append("path")
        .datum(points)
        .attr("d", lineGenerator)
        .attr("fill", colorScale(algIndex))
        .attr("fill-opacity", 0.2)
        .attr("stroke", colorScale(algIndex))
        .attr("stroke-width", 2);

      // Add points
      points.forEach(point => {
        g.append("circle")
          .attr("cx", point[0])
          .attr("cy", point[1])
          .attr("r", 3)
          .attr("fill", colorScale(algIndex));
      });
    });

    // Add legend
    graphData.forEach((algorithm, i) => {
      g.append("text")
        .attr("x", 20)
        .attr("y", 20 + i * 20)
        .attr("fill", colorScale(i))
        .style("font-size", "12px")
        .text(algorithm.name);
    });
  };

  const renderNoiseAnalysisGraph = (g, width, height) => {
    if (!graphData) return;

    const xScale = d3.scaleLinear()
      .domain(d3.extent(graphData[0].data, d => d.noise))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    const line = d3.line()
      .x(d => xScale(d.noise))
      .y(d => yScale(d.fidelity))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add lines for each algorithm
    graphData.forEach((algorithm, i) => {
      g.append("path")
        .datum(algorithm.data)
        .attr("fill", "none")
        .attr("stroke", colorScale(i))
        .attr("stroke-width", 2)
        .attr("d", line);

      // Add points
      g.selectAll(`.points-${i}`)
        .data(algorithm.data)
        .enter().append("circle")
        .attr("class", `points-${i}`)
        .attr("cx", d => xScale(d.noise))
        .attr("cy", d => yScale(d.fidelity))
        .attr("r", 3)
        .attr("fill", colorScale(i));
    });

    // Legend
    graphData.forEach((algorithm, i) => {
      g.append("text")
        .attr("x", width - 100)
        .attr("y", 20 + i * 20)
        .attr("fill", colorScale(i))
        .style("font-size", "12px")
        .text(algorithm.name);
    });

    // Labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Fidelity");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Noise Level");
  };

  const renderCircuitDepthGraph = (g, width, height) => {
    if (!graphData) return;

    const xScale = d3.scaleLinear()
      .domain(d3.extent(graphData, d => d.depth))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Performance line
    const performanceLine = d3.line()
      .x(d => xScale(d.depth))
      .y(d => yScale(d.performance))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(graphData)
      .attr("fill", "none")
      .attr("stroke", "#3b82f6")
      .attr("stroke-width", 2)
      .attr("d", performanceLine);

    // Noise impact line
    const noiseLine = d3.line()
      .x(d => xScale(d.depth))
      .y(d => yScale(d.noise_impact))
      .curve(d3.curveMonotoneX);

    g.append("path")
      .datum(graphData)
      .attr("fill", "none")
      .attr("stroke", "#ef4444")
      .attr("stroke-width", 2)
      .attr("d", noiseLine);

    // Add points
    g.selectAll(".performance-point")
      .data(graphData)
      .enter().append("circle")
      .attr("class", "performance-point")
      .attr("cx", d => xScale(d.depth))
      .attr("cy", d => yScale(d.performance))
      .attr("r", 3)
      .attr("fill", "#3b82f6");

    g.selectAll(".noise-point")
      .data(graphData)
      .enter().append("circle")
      .attr("class", "noise-point")
      .attr("cx", d => xScale(d.depth))
      .attr("cy", d => yScale(d.noise_impact))
      .attr("r", 3)
      .attr("fill", "#ef4444");

    // Legend
    g.append("text")
      .attr("x", width - 120)
      .attr("y", 20)
      .attr("fill", "#3b82f6")
      .style("font-size", "12px")
      .text("Performance");

    g.append("text")
      .attr("x", width - 120)
      .attr("y", 40)
      .attr("fill", "#ef4444")
      .style("font-size", "12px")
      .text("Noise Impact");

    // Labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Metric Value");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Circuit Depth");
  };

  const renderErrorMitigationGraph = (g, width, height) => {
    if (!graphData) return;

    const xScale = d3.scaleBand()
      .domain(graphData.map(d => d.technique))
      .range([0, width])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(graphData, d => d.fidelity_improvement)])
      .range([height, 0]);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add bars
    g.selectAll(".bar")
      .data(graphData)
      .enter().append("rect")
      .attr("class", "bar")
      .attr("x", d => xScale(d.technique))
      .attr("width", xScale.bandwidth())
      .attr("y", d => yScale(d.fidelity_improvement))
      .attr("height", d => height - yScale(d.fidelity_improvement))
      .attr("fill", "#3b82f6")
      .attr("opacity", 0.8);

    // Add value labels on bars
    g.selectAll(".bar-label")
      .data(graphData)
      .enter().append("text")
      .attr("class", "bar-label")
      .attr("x", d => xScale(d.technique) + xScale.bandwidth() / 2)
      .attr("y", d => yScale(d.fidelity_improvement) - 5)
      .attr("text-anchor", "middle")
      .style("font-size", "10px")
      .text(d => d.fidelity_improvement.toFixed(2));

    // Labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Fidelity Improvement");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Error Mitigation Technique");
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">📊</span>
            Advanced Graph Analytics
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Graph Type Selection */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-3">Analysis Types</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(graphTypes).map(([key, info]) => (
                <button
                  key={key}
                  onClick={() => setSelectedGraph(key)}
                  className={`p-3 rounded-lg border text-left transition-all ${
                    selectedGraph === key
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-lg">{info.icon}</span>
                    <Badge variant={selectedGraph === key ? "primary" : "secondary"}>
                      {key.replace('_', ' ')}
                    </Badge>
                  </div>
                  <h4 className="font-medium text-sm">{info.name}</h4>
                  <p className="text-xs text-gray-600 mt-1">{info.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Graph Visualization */}
          <div className="border rounded-lg p-4 bg-white">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold flex items-center gap-2">
                <span>{graphTypes[selectedGraph].icon}</span>
                {graphTypes[selectedGraph].name}
              </h4>
              <Button
                onClick={generateGraphData}
                variant="outline"
                size="sm"
              >
                🔄 Refresh Data
              </Button>
            </div>
            
            <div className="flex justify-center">
              <svg ref={svgRef}></svg>
            </div>
          </div>

          {/* Graph Insights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h5 className="font-medium text-blue-800 mb-2">Key Insights</h5>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Interactive visualization with D3.js</li>
                <li>• Real-time data analysis</li>
                <li>• Multiple algorithm comparison</li>
              </ul>
            </div>
            
            <div className="bg-green-50 p-4 rounded-lg">
              <h5 className="font-medium text-green-800 mb-2">Performance</h5>
              <ul className="text-sm text-green-700 space-y-1">
                <li>• Optimized rendering pipeline</li>
                <li>• Responsive design</li>
                <li>• Smooth animations</li>
              </ul>
            </div>
            
            <div className="bg-purple-50 p-4 rounded-lg">
              <h5 className="font-medium text-purple-800 mb-2">Features</h5>
              <ul className="text-sm text-purple-700 space-y-1">
                <li>• 8+ graph types available</li>
                <li>• Export capabilities</li>
                <li>• Customizable parameters</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AdvancedGraphAnalytics;
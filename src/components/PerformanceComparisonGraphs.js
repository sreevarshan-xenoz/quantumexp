import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

const PerformanceComparisonGraphs = ({ comparisonData }) => {
  const [selectedComparison, setSelectedComparison] = useState('accuracy_vs_time');
  const [animationEnabled, setAnimationEnabled] = useState(true);
  const svgRef = useRef();

  const comparisonTypes = {
    accuracy_vs_time: {
      name: 'Accuracy vs Training Time',
      icon: '⏱️',
      description: 'Compare model accuracy against training time'
    },
    scalability_analysis: {
      name: 'Scalability Analysis',
      icon: '📈',
      description: 'Performance scaling with problem size'
    },
    resource_efficiency: {
      name: 'Resource Efficiency',
      icon: '⚡',
      description: 'Memory and compute resource utilization'
    },
    noise_robustness: {
      name: 'Noise Robustness',
      icon: '🌊',
      description: 'Performance under different noise conditions'
    },
    quantum_classical_comparison: {
      name: 'Quantum vs Classical',
      icon: '⚖️',
      description: 'Direct quantum-classical algorithm comparison'
    },
    convergence_speed: {
      name: 'Convergence Speed',
      icon: '🎯',
      description: 'Optimization convergence characteristics'
    },
    cost_benefit_analysis: {
      name: 'Cost-Benefit Analysis',
      icon: '💰',
      description: 'Performance per dollar spent'
    },
    multi_metric_radar: {
      name: 'Multi-Metric Radar',
      icon: '🕸️',
      description: 'Comprehensive multi-dimensional comparison'
    }
  };

  useEffect(() => {
    if (comparisonData || selectedComparison) {
      renderComparison();
    }
  }, [selectedComparison, comparisonData, animationEnabled]);

  const generateSyntheticData = () => {
    const algorithms = [
      { name: 'VQC', type: 'quantum', color: '#3b82f6' },
      { name: 'QAOA', type: 'quantum', color: '#10b981' },
      { name: 'VQE', type: 'quantum', color: '#8b5cf6' },
      { name: 'Random Forest', type: 'classical', color: '#f59e0b' },
      { name: 'SVM', type: 'classical', color: '#ef4444' },
      { name: 'Neural Network', type: 'classical', color: '#06b6d4' }
    ];

    switch (selectedComparison) {
      case 'accuracy_vs_time':
        return algorithms.map(alg => ({
          ...alg,
          accuracy: 0.7 + Math.random() * 0.25,
          time: alg.type === 'quantum' ? 10 + Math.random() * 20 : 5 + Math.random() * 15,
          variance: 0.02 + Math.random() * 0.03
        }));

      case 'scalability_analysis':
        return algorithms.map(alg => ({
          ...alg,
          data: Array.from({ length: 8 }, (_, i) => {
            const size = Math.pow(2, i + 1);
            const baseTime = alg.type === 'quantum' ? size * 0.5 : size * 1.2;
            return {
              problemSize: size,
              executionTime: baseTime + Math.random() * baseTime * 0.3,
              accuracy: Math.max(0.5, 0.95 - (size * 0.01) + Math.random() * 0.1)
            };
          })
        }));

      case 'resource_efficiency':
        return algorithms.map(alg => ({
          ...alg,
          memory: alg.type === 'quantum' ? 50 + Math.random() * 100 : 100 + Math.random() * 200,
          cpu: alg.type === 'quantum' ? 30 + Math.random() * 40 : 60 + Math.random() * 80,
          efficiency: (Math.random() * 0.5 + 0.5) * (alg.type === 'quantum' ? 1.2 : 1.0)
        }));

      case 'noise_robustness':
        return algorithms.map(alg => ({
          ...alg,
          data: Array.from({ length: 10 }, (_, i) => {
            const noiseLevel = i * 0.05;
            const baseFidelity = alg.type === 'quantum' ? 0.9 : 0.95;
            return {
              noiseLevel,
              fidelity: Math.max(0.1, baseFidelity - noiseLevel * (2 + Math.random())),
              performance: Math.max(0.1, 0.9 - noiseLevel * (1.5 + Math.random() * 0.5))
            };
          })
        }));

      case 'quantum_classical_comparison':
        return {
          quantum: {
            advantages: ['Exponential speedup potential', 'Natural quantum problem solving', 'Parallel computation'],
            disadvantages: ['Hardware limitations', 'Noise sensitivity', 'Limited qubit count'],
            metrics: { speed: 0.8, accuracy: 0.75, scalability: 0.9, robustness: 0.6 }
          },
          classical: {
            advantages: ['Mature hardware', 'Noise-free computation', 'Extensive libraries'],
            disadvantages: ['Exponential scaling issues', 'Limited parallelism', 'Classical complexity limits'],
            metrics: { speed: 0.7, accuracy: 0.85, scalability: 0.6, robustness: 0.9 }
          }
        };

      case 'convergence_speed':
        return algorithms.map(alg => ({
          ...alg,
          data: Array.from({ length: 50 }, (_, i) => ({
            iteration: i,
            loss: Math.exp(-i / (10 + Math.random() * 10)) + Math.random() * 0.1,
            gradient: Math.exp(-i / 15) * (1 + Math.random() * 0.5)
          }))
        }));

      case 'cost_benefit_analysis':
        return algorithms.map(alg => ({
          ...alg,
          cost: alg.type === 'quantum' ? 50 + Math.random() * 100 : 10 + Math.random() * 30,
          benefit: 0.6 + Math.random() * 0.3,
          roi: (0.6 + Math.random() * 0.3) / (alg.type === 'quantum' ? 0.8 : 1.2)
        }));

      case 'multi_metric_radar':
        return algorithms.map(alg => ({
          ...alg,
          metrics: {
            accuracy: 0.6 + Math.random() * 0.3,
            speed: 0.5 + Math.random() * 0.4,
            scalability: 0.4 + Math.random() * 0.5,
            robustness: 0.5 + Math.random() * 0.4,
            interpretability: alg.type === 'classical' ? 0.7 + Math.random() * 0.2 : 0.3 + Math.random() * 0.3,
            cost_efficiency: alg.type === 'classical' ? 0.8 + Math.random() * 0.2 : 0.4 + Math.random() * 0.4
          }
        }));

      default:
        return [];
    }
  };

  const renderComparison = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const data = generateSyntheticData();
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    switch (selectedComparison) {
      case 'accuracy_vs_time':
        renderAccuracyVsTime(g, data, width, height);
        break;
      case 'scalability_analysis':
        renderScalabilityAnalysis(g, data, width, height);
        break;
      case 'resource_efficiency':
        renderResourceEfficiency(g, data, width, height);
        break;
      case 'noise_robustness':
        renderNoiseRobustness(g, data, width, height);
        break;
      case 'quantum_classical_comparison':
        renderQuantumClassicalComparison(g, data, width, height);
        break;
      case 'convergence_speed':
        renderConvergenceSpeed(g, data, width, height);
        break;
      case 'cost_benefit_analysis':
        renderCostBenefitAnalysis(g, data, width, height);
        break;
      case 'multi_metric_radar':
        renderMultiMetricRadar(g, data, width, height);
        break;
    }
  };

  const renderAccuracyVsTime = (g, data, width, height) => {
    const xScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.time))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.accuracy))
      .range([height, 0]);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add bubbles
    const bubbles = g.selectAll(".bubble")
      .data(data)
      .enter().append("circle")
      .attr("class", "bubble")
      .attr("cx", d => xScale(d.time))
      .attr("cy", d => yScale(d.accuracy))
      .attr("r", 0)
      .attr("fill", d => d.color)
      .attr("opacity", 0.7)
      .attr("stroke", "white")
      .attr("stroke-width", 2);

    if (animationEnabled) {
      bubbles.transition()
        .duration(1000)
        .attr("r", d => 5 + d.variance * 100);
    } else {
      bubbles.attr("r", d => 5 + d.variance * 100);
    }

    // Add labels
    g.selectAll(".label")
      .data(data)
      .enter().append("text")
      .attr("class", "label")
      .attr("x", d => xScale(d.time))
      .attr("y", d => yScale(d.accuracy) - 15)
      .attr("text-anchor", "middle")
      .style("font-size", "10px")
      .style("font-weight", "bold")
      .text(d => d.name);

    // Add axis labels
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
      .text("Training Time (minutes)");
  };

  const renderScalabilityAnalysis = (g, data, width, height) => {
    const xScale = d3.scaleLog()
      .domain([1, 256])
      .range([0, width]);

    const yScale = d3.scaleLog()
      .domain([0.1, 1000])
      .range([height, 0]);

    const line = d3.line()
      .x(d => xScale(d.problemSize))
      .y(d => yScale(d.executionTime))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.format("d")));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add lines for each algorithm
    data.forEach((algorithm, i) => {
      const path = g.append("path")
        .datum(algorithm.data)
        .attr("fill", "none")
        .attr("stroke", algorithm.color)
        .attr("stroke-width", 3)
        .attr("d", line);

      if (animationEnabled) {
        const totalLength = path.node().getTotalLength();
        path
          .attr("stroke-dasharray", totalLength + " " + totalLength)
          .attr("stroke-dashoffset", totalLength)
          .transition()
          .duration(2000)
          .ease(d3.easeLinear)
          .attr("stroke-dashoffset", 0);
      }

      // Add algorithm label
      g.append("text")
        .attr("x", width - 100)
        .attr("y", 20 + i * 20)
        .attr("fill", algorithm.color)
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text(algorithm.name);
    });

    // Add axis labels
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
      .text("Problem Size (log scale)");
  };

  const renderResourceEfficiency = (g, data, width, height) => {
    const xScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.memory))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.cpu))
      .range([height, 0]);

    const sizeScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.efficiency))
      .range([5, 25]);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add bubbles
    const bubbles = g.selectAll(".efficiency-bubble")
      .data(data)
      .enter().append("circle")
      .attr("class", "efficiency-bubble")
      .attr("cx", d => xScale(d.memory))
      .attr("cy", d => yScale(d.cpu))
      .attr("r", 0)
      .attr("fill", d => d.color)
      .attr("opacity", 0.6)
      .attr("stroke", "white")
      .attr("stroke-width", 2);

    if (animationEnabled) {
      bubbles.transition()
        .duration(1500)
        .attr("r", d => sizeScale(d.efficiency));
    } else {
      bubbles.attr("r", d => sizeScale(d.efficiency));
    }

    // Add labels
    g.selectAll(".efficiency-label")
      .data(data)
      .enter().append("text")
      .attr("class", "efficiency-label")
      .attr("x", d => xScale(d.memory))
      .attr("y", d => yScale(d.cpu) - sizeScale(d.efficiency) - 5)
      .attr("text-anchor", "middle")
      .style("font-size", "10px")
      .style("font-weight", "bold")
      .text(d => d.name);

    // Add axis labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("CPU Usage (%)");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Memory Usage (MB)");
  };

  const renderMultiMetricRadar = (g, data, width, height) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 50;

    const metrics = Object.keys(data[0].metrics);
    const angleScale = d3.scaleLinear()
      .domain([0, metrics.length])
      .range([0, 2 * Math.PI]);

    const radiusScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, radius]);

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
    metrics.forEach((metric, i) => {
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
        .style("font-size", "11px")
        .style("font-weight", "bold")
        .text(metric.replace('_', ' '));
    });

    // Draw algorithm polygons
    data.forEach((algorithm, algIndex) => {
      const points = metrics.map((metric, i) => {
        const angle = angleScale(i) - Math.PI / 2;
        const r = radiusScale(algorithm.metrics[metric]);
        return [
          centerX + Math.cos(angle) * r,
          centerY + Math.sin(angle) * r
        ];
      });

      const lineGenerator = d3.line()
        .x(d => d[0])
        .y(d => d[1])
        .curve(d3.curveLinearClosed);

      const path = g.append("path")
        .datum(points)
        .attr("d", lineGenerator)
        .attr("fill", algorithm.color)
        .attr("fill-opacity", 0.2)
        .attr("stroke", algorithm.color)
        .attr("stroke-width", 2);

      if (animationEnabled) {
        const totalLength = path.node().getTotalLength();
        path
          .attr("stroke-dasharray", totalLength + " " + totalLength)
          .attr("stroke-dashoffset", totalLength)
          .transition()
          .duration(1500)
          .delay(algIndex * 200)
          .ease(d3.easeLinear)
          .attr("stroke-dashoffset", 0);
      }

      // Add points
      points.forEach(point => {
        g.append("circle")
          .attr("cx", point[0])
          .attr("cy", point[1])
          .attr("r", 0)
          .attr("fill", algorithm.color)
          .transition()
          .duration(1000)
          .delay(algIndex * 200)
          .attr("r", 3);
      });
    });

    // Add legend
    data.forEach((algorithm, i) => {
      g.append("text")
        .attr("x", 20)
        .attr("y", 20 + i * 20)
        .attr("fill", algorithm.color)
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text(algorithm.name);
    });
  };

  const renderCostBenefitAnalysis = (g, data, width, height) => {
    const xScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.cost))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.benefit))
      .range([height, 0]);

    const sizeScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.roi))
      .range([8, 30]);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add quadrant lines
    const avgCost = d3.mean(data, d => d.cost);
    const avgBenefit = d3.mean(data, d => d.benefit);

    g.append("line")
      .attr("x1", xScale(avgCost))
      .attr("y1", 0)
      .attr("x2", xScale(avgCost))
      .attr("y2", height)
      .attr("stroke", "#ccc")
      .attr("stroke-dasharray", "5,5");

    g.append("line")
      .attr("x1", 0)
      .attr("y1", yScale(avgBenefit))
      .attr("x2", width)
      .attr("y2", yScale(avgBenefit))
      .attr("stroke", "#ccc")
      .attr("stroke-dasharray", "5,5");

    // Add bubbles
    const bubbles = g.selectAll(".cost-bubble")
      .data(data)
      .enter().append("circle")
      .attr("class", "cost-bubble")
      .attr("cx", d => xScale(d.cost))
      .attr("cy", d => yScale(d.benefit))
      .attr("r", 0)
      .attr("fill", d => d.color)
      .attr("opacity", 0.7)
      .attr("stroke", "white")
      .attr("stroke-width", 2);

    if (animationEnabled) {
      bubbles.transition()
        .duration(1200)
        .attr("r", d => sizeScale(d.roi));
    } else {
      bubbles.attr("r", d => sizeScale(d.roi));
    }

    // Add labels
    g.selectAll(".cost-label")
      .data(data)
      .enter().append("text")
      .attr("class", "cost-label")
      .attr("x", d => xScale(d.cost))
      .attr("y", d => yScale(d.benefit) - sizeScale(d.roi) - 5)
      .attr("text-anchor", "middle")
      .style("font-size", "10px")
      .style("font-weight", "bold")
      .text(d => d.name);

    // Add quadrant labels
    g.append("text")
      .attr("x", width - 80)
      .attr("y", 20)
      .style("font-size", "12px")
      .style("fill", "#666")
      .text("High Cost, High Benefit");

    g.append("text")
      .attr("x", 10)
      .attr("y", 20)
      .style("font-size", "12px")
      .style("fill", "#666")
      .text("Low Cost, High Benefit");

    // Add axis labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Benefit Score");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Cost ($)");
  };

  const renderConvergenceSpeed = (g, data, width, height) => {
    const xScale = d3.scaleLinear()
      .domain([0, 49])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    const line = d3.line()
      .x(d => xScale(d.iteration))
      .y(d => yScale(d.loss))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add lines for each algorithm
    data.forEach((algorithm, i) => {
      const path = g.append("path")
        .datum(algorithm.data)
        .attr("fill", "none")
        .attr("stroke", algorithm.color)
        .attr("stroke-width", 2)
        .attr("d", line);

      if (animationEnabled) {
        const totalLength = path.node().getTotalLength();
        path
          .attr("stroke-dasharray", totalLength + " " + totalLength)
          .attr("stroke-dashoffset", totalLength)
          .transition()
          .duration(3000)
          .delay(i * 300)
          .ease(d3.easeLinear)
          .attr("stroke-dashoffset", 0);
      }

      // Add algorithm label
      g.append("text")
        .attr("x", width - 100)
        .attr("y", 20 + i * 20)
        .attr("fill", algorithm.color)
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text(algorithm.name);
    });

    // Add axis labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Loss");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Iteration");
  };

  const renderNoiseRobustness = (g, data, width, height) => {
    const xScale = d3.scaleLinear()
      .domain([0, 0.45])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    const line = d3.line()
      .x(d => xScale(d.noiseLevel))
      .y(d => yScale(d.fidelity))
      .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add lines for each algorithm
    data.forEach((algorithm, i) => {
      const path = g.append("path")
        .datum(algorithm.data)
        .attr("fill", "none")
        .attr("stroke", algorithm.color)
        .attr("stroke-width", 3)
        .attr("d", line);

      if (animationEnabled) {
        const totalLength = path.node().getTotalLength();
        path
          .attr("stroke-dasharray", totalLength + " " + totalLength)
          .attr("stroke-dashoffset", totalLength)
          .transition()
          .duration(2000)
          .delay(i * 200)
          .ease(d3.easeLinear)
          .attr("stroke-dashoffset", 0);
      }

      // Add points
      g.selectAll(`.noise-points-${i}`)
        .data(algorithm.data)
        .enter().append("circle")
        .attr("class", `noise-points-${i}`)
        .attr("cx", d => xScale(d.noiseLevel))
        .attr("cy", d => yScale(d.fidelity))
        .attr("r", 0)
        .attr("fill", algorithm.color)
        .transition()
        .duration(1000)
        .delay(i * 200)
        .attr("r", 3);

      // Add algorithm label
      g.append("text")
        .attr("x", width - 100)
        .attr("y", 20 + i * 20)
        .attr("fill", algorithm.color)
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text(algorithm.name);
    });

    // Add axis labels
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

  const renderQuantumClassicalComparison = (g, data, width, height) => {
    // This would render a more complex comparison visualization
    // For now, we'll create a simple bar chart comparison
    const metrics = Object.keys(data.quantum.metrics);
    const barWidth = width / (metrics.length * 2 + 1);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(d3.scaleBand().domain(metrics).range([0, width])));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add bars for quantum
    metrics.forEach((metric, i) => {
      const x = (i * 2 + 0.5) * barWidth;
      
      g.append("rect")
        .attr("x", x)
        .attr("y", height)
        .attr("width", barWidth * 0.8)
        .attr("height", 0)
        .attr("fill", "#3b82f6")
        .transition()
        .duration(1000)
        .delay(i * 100)
        .attr("y", yScale(data.quantum.metrics[metric]))
        .attr("height", height - yScale(data.quantum.metrics[metric]));

      // Add bars for classical
      g.append("rect")
        .attr("x", x + barWidth)
        .attr("y", height)
        .attr("width", barWidth * 0.8)
        .attr("height", 0)
        .attr("fill", "#ef4444")
        .transition()
        .duration(1000)
        .delay(i * 100 + 50)
        .attr("y", yScale(data.classical.metrics[metric]))
        .attr("height", height - yScale(data.classical.metrics[metric]));

      // Add metric labels
      g.append("text")
        .attr("x", x + barWidth)
        .attr("y", height + 20)
        .attr("text-anchor", "middle")
        .style("font-size", "10px")
        .text(metric);
    });

    // Add legend
    g.append("rect")
      .attr("x", width - 120)
      .attr("y", 20)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "#3b82f6");

    g.append("text")
      .attr("x", width - 100)
      .attr("y", 32)
      .style("font-size", "12px")
      .text("Quantum");

    g.append("rect")
      .attr("x", width - 120)
      .attr("y", 40)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "#ef4444");

    g.append("text")
      .attr("x", width - 100)
      .attr("y", 52)
      .style("font-size", "12px")
      .text("Classical");
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">📊</span>
            Performance Comparison Graphs
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Comparison Type Selection */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold">Comparison Types</h3>
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={animationEnabled}
                    onChange={(e) => setAnimationEnabled(e.target.checked)}
                  />
                  <span className="text-sm">Enable Animations</span>
                </label>
              </div>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(comparisonTypes).map(([key, info]) => (
                <button
                  key={key}
                  onClick={() => setSelectedComparison(key)}
                  className={`p-3 rounded-lg border text-left transition-all ${
                    selectedComparison === key
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-lg">{info.icon}</span>
                    <Badge variant={selectedComparison === key ? "primary" : "secondary"}>
                      {key.split('_')[0]}
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
                <span>{comparisonTypes[selectedComparison].icon}</span>
                {comparisonTypes[selectedComparison].name}
              </h4>
              <Button
                onClick={renderComparison}
                variant="outline"
                size="sm"
              >
                🔄 Regenerate
              </Button>
            </div>
            
            <div className="flex justify-center">
              <svg ref={svgRef}></svg>
            </div>
          </div>

          {/* Performance Insights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg">
              <h5 className="font-medium text-blue-800 mb-2">Quantum Algorithms</h5>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• VQC: Variational classification</li>
                <li>• QAOA: Optimization problems</li>
                <li>• VQE: Ground state energy</li>
              </ul>
            </div>
            
            <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg">
              <h5 className="font-medium text-green-800 mb-2">Classical Algorithms</h5>
              <ul className="text-sm text-green-700 space-y-1">
                <li>• Random Forest: Ensemble method</li>
                <li>• SVM: Support vector machines</li>
                <li>• Neural Network: Deep learning</li>
              </ul>
            </div>
            
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
              <h5 className="font-medium text-purple-800 mb-2">Analysis Features</h5>
              <ul className="text-sm text-purple-700 space-y-1">
                <li>• Interactive D3.js visualizations</li>
                <li>• Real-time performance comparison</li>
                <li>• Multi-dimensional analysis</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PerformanceComparisonGraphs;
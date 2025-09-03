import React, { useState, useMemo, useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

const DatasetComparison = ({ datasets, onDatasetSelect }) => {
  const [selectedDatasets, setSelectedDatasets] = useState(['circles', 'moons']);
  const [comparisonMetric, setComparisonMetric] = useState('separability');
  const radarRef = useRef();
  const barRef = useRef();

  // Calculate dataset metrics
  const datasetMetrics = useMemo(() => {
    const metrics = {};
    
    Object.entries(datasets).forEach(([name, dataset]) => {
      if (!dataset || !dataset.X || !dataset.y) return;
      
      const { X, y } = dataset;
      
      // Calculate separability (distance between class centroids)
      const class0Points = X.filter((_, i) => y[i] === 0);
      const class1Points = X.filter((_, i) => y[i] === 1);
      
      let separability = 0;
      if (class0Points.length > 0 && class1Points.length > 0) {
        const centroid0 = class0Points.reduce((acc, point) => 
          acc.map((val, idx) => val + point[idx]), 
          new Array(X[0].length).fill(0)
        ).map(val => val / class0Points.length);
        
        const centroid1 = class1Points.reduce((acc, point) => 
          acc.map((val, idx) => val + point[idx]), 
          new Array(X[0].length).fill(0)
        ).map(val => val / class1Points.length);
        
        separability = Math.sqrt(
          centroid0.reduce((sum, val, idx) => 
            sum + Math.pow(val - centroid1[idx], 2), 0
          )
        );
      }
      
      // Calculate complexity (average feature variance)
      const complexity = X[0].map((_, featureIdx) => {
        const values = X.map(point => point[featureIdx]);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      }).reduce((a, b) => a + b, 0) / X[0].length;
      
      // Calculate class balance
      const classBalance = Math.min(
        class0Points.length / X.length,
        class1Points.length / X.length
      ) * 2; // Normalize to 0-1 where 1 is perfectly balanced
      
      // Calculate dimensionality
      const dimensionality = X[0].length;
      
      // Calculate density (points per unit area/volume)
      const ranges = X[0].map((_, featureIdx) => {
        const values = X.map(point => point[featureIdx]);
        return Math.max(...values) - Math.min(...values);
      });
      const volume = ranges.reduce((a, b) => a * b, 1);
      const density = X.length / volume;
      
      metrics[name] = {
        separability: Math.min(separability, 5) / 5, // Normalize to 0-1
        complexity: Math.min(complexity, 2) / 2, // Normalize to 0-1
        classBalance,
        dimensionality: Math.min(dimensionality, 10) / 10, // Normalize to 0-1
        density: Math.min(density, 1000) / 1000, // Normalize to 0-1
        size: X.length
      };
    });
    
    return metrics;
  }, [datasets]);

  // Render radar chart
  useEffect(() => {
    if (!radarRef.current || selectedDatasets.length === 0) return;

    const svg = d3.select(radarRef.current);
    svg.selectAll("*").remove();

    const width = 400;
    const height = 400;
    const margin = 50;
    const radius = Math.min(width, height) / 2 - margin;
    const centerX = width / 2;
    const centerY = height / 2;

    const metrics = ['separability', 'complexity', 'classBalance', 'dimensionality', 'density'];
    const angleScale = d3.scaleLinear()
      .domain([0, metrics.length])
      .range([0, 2 * Math.PI]);

    const radiusScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, radius]);

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Draw radar grid
    const levels = 5;
    for (let i = 1; i <= levels; i++) {
      const levelRadius = (radius / levels) * i;
      
      svg.append("circle")
        .attr("cx", centerX)
        .attr("cy", centerY)
        .attr("r", levelRadius)
        .attr("fill", "none")
        .attr("stroke", "#ddd")
        .attr("stroke-width", 1);
      
      // Add level labels
      svg.append("text")
        .attr("x", centerX + levelRadius + 5)
        .attr("y", centerY)
        .attr("text-anchor", "start")
        .attr("dominant-baseline", "middle")
        .style("font-size", "10px")
        .style("fill", "#666")
        .text((i / levels).toFixed(1));
    }

    // Draw axes
    metrics.forEach((metric, i) => {
      const angle = angleScale(i) - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      svg.append("line")
        .attr("x1", centerX)
        .attr("y1", centerY)
        .attr("x2", x)
        .attr("y2", y)
        .attr("stroke", "#ddd")
        .attr("stroke-width", 1);

      // Add metric labels
      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 20);
      
      svg.append("text")
        .attr("x", labelX)
        .attr("y", labelY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "12px")
        .style("font-weight", "bold")
        .text(metric.charAt(0).toUpperCase() + metric.slice(1));
    });

    // Draw dataset polygons
    selectedDatasets.forEach((datasetName, datasetIndex) => {
      const datasetData = datasetMetrics[datasetName];
      if (!datasetData) return;

      const points = metrics.map((metric, i) => {
        const angle = angleScale(i) - Math.PI / 2;
        const r = radiusScale(datasetData[metric] || 0);
        return [
          centerX + Math.cos(angle) * r,
          centerY + Math.sin(angle) * r
        ];
      });

      const lineGenerator = d3.line()
        .x(d => d[0])
        .y(d => d[1])
        .curve(d3.curveLinearClosed);

      // Add polygon
      svg.append("path")
        .datum(points)
        .attr("d", lineGenerator)
        .attr("fill", colorScale(datasetIndex))
        .attr("fill-opacity", 0.2)
        .attr("stroke", colorScale(datasetIndex))
        .attr("stroke-width", 2);

      // Add points
      points.forEach(point => {
        svg.append("circle")
          .attr("cx", point[0])
          .attr("cy", point[1])
          .attr("r", 3)
          .attr("fill", colorScale(datasetIndex));
      });
    });

    // Add legend
    selectedDatasets.forEach((datasetName, i) => {
      svg.append("circle")
        .attr("cx", 20)
        .attr("cy", 20 + i * 20)
        .attr("r", 6)
        .attr("fill", colorScale(i));

      svg.append("text")
        .attr("x", 35)
        .attr("y", 20 + i * 20)
        .attr("dominant-baseline", "middle")
        .style("font-size", "12px")
        .text(datasetName.charAt(0).toUpperCase() + datasetName.slice(1));
    });
  }, [datasetMetrics, selectedDatasets]);

  // Render comparison bar chart
  useEffect(() => {
    if (!barRef.current || selectedDatasets.length === 0) return;

    const svg = d3.select(barRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 500 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const data = selectedDatasets.map(name => ({
      name,
      value: datasetMetrics[name]?.[comparisonMetric] || 0
    }));

    const xScale = d3.scaleBand()
      .domain(data.map(d => d.name))
      .range([0, width])
      .padding(0.1);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value)])
      .range([height, 0]);

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add bars
    g.selectAll(".bar")
      .data(data)
      .enter().append("rect")
      .attr("class", "bar")
      .attr("x", d => xScale(d.name))
      .attr("width", xScale.bandwidth())
      .attr("y", d => yScale(d.value))
      .attr("height", d => height - yScale(d.value))
      .attr("fill", (d, i) => colorScale(i))
      .attr("opacity", 0.8);

    // Add value labels
    g.selectAll(".label")
      .data(data)
      .enter().append("text")
      .attr("class", "label")
      .attr("x", d => xScale(d.name) + xScale.bandwidth() / 2)
      .attr("y", d => yScale(d.value) - 5)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .style("font-weight", "bold")
      .text(d => d.value.toFixed(3));

    // Add axis labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text(comparisonMetric.charAt(0).toUpperCase() + comparisonMetric.slice(1));

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Dataset");
  }, [datasetMetrics, selectedDatasets, comparisonMetric]);

  const availableDatasets = Object.keys(datasets);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">⚖️</span>
            Dataset Comparison
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Controls */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Datasets to Compare
              </label>
              <div className="space-y-2">
                {availableDatasets.map(name => (
                  <label key={name} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={selectedDatasets.includes(name)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedDatasets([...selectedDatasets, name]);
                        } else {
                          setSelectedDatasets(selectedDatasets.filter(d => d !== name));
                        }
                      }}
                      className="rounded"
                    />
                    <span className="capitalize">{name}</span>
                    <Badge variant="outline">
                      {datasets[name]?.X?.length || 0} samples
                    </Badge>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Comparison Metric
              </label>
              <select
                value={comparisonMetric}
                onChange={(e) => setComparisonMetric(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="separability">Class Separability</option>
                <option value="complexity">Data Complexity</option>
                <option value="classBalance">Class Balance</option>
                <option value="dimensionality">Dimensionality</option>
                <option value="density">Data Density</option>
              </select>
            </div>
          </div>

          {/* Visualizations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="text-lg font-semibold mb-4">Multi-Metric Radar Chart</h4>
              <div className="flex justify-center">
                <svg ref={radarRef} width="400" height="400"></svg>
              </div>
            </div>

            <div>
              <h4 className="text-lg font-semibold mb-4">
                {comparisonMetric.charAt(0).toUpperCase() + comparisonMetric.slice(1)} Comparison
              </h4>
              <div className="flex justify-center">
                <svg ref={barRef}></svg>
              </div>
            </div>
          </div>

          {/* Detailed Metrics Table */}
          <div className="mt-6">
            <h4 className="text-lg font-semibold mb-4">Detailed Metrics</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Dataset
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Separability
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Complexity
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Class Balance
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Dimensionality
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Size
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Action
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {selectedDatasets.map((name, index) => {
                    const metrics = datasetMetrics[name];
                    if (!metrics) return null;

                    return (
                      <tr key={name} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 capitalize">
                          {name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {metrics.separability.toFixed(3)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {metrics.complexity.toFixed(3)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {metrics.classBalance.toFixed(3)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {metrics.dimensionality.toFixed(3)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {metrics.size}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <Button
                            size="sm"
                            onClick={() => onDatasetSelect && onDatasetSelect(datasets[name], name)}
                          >
                            Select
                          </Button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Recommendations */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h5 className="font-medium text-blue-800 mb-2">🎯 For Beginners</h5>
              <p className="text-sm text-blue-700">
                Start with <strong>Circles</strong> or <strong>Blobs</strong> datasets. 
                They have high separability and are easier to classify.
              </p>
            </div>

            <div className="bg-orange-50 p-4 rounded-lg">
              <h5 className="font-medium text-orange-800 mb-2">🔥 For Challenge</h5>
              <p className="text-sm text-orange-700">
                Try <strong>Moons</strong> with high noise or <strong>Classification</strong> 
                datasets for more complex decision boundaries.
              </p>
            </div>

            <div className="bg-green-50 p-4 rounded-lg">
              <h5 className="font-medium text-green-800 mb-2">⚛️ For Quantum Advantage</h5>
              <p className="text-sm text-green-700">
                Use datasets with high dimensionality and complex patterns 
                where quantum algorithms might show advantages.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default DatasetComparison;
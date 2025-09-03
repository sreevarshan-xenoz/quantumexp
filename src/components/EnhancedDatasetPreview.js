import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as d3 from 'd3';
import * as THREE from 'three';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';

const EnhancedDatasetPreview = ({ onDatasetSelect, onParametersChange }) => {
  const [activeTab, setActiveTab] = useState('3d');
  const [datasetType, setDatasetType] = useState('circles');
  const [noiseLevel, setNoiseLevel] = useState(0.2);
  const [sampleSize, setSampleSize] = useState(500);
  const [selectedFeature, setSelectedFeature] = useState(0);
  const [classFilter, setClassFilter] = useState('all');
  const [datasets, setDatasets] = useState({});
  const [currentDataset, setCurrentDataset] = useState(null);
  
  const threejsRef = useRef();
  const scatterRef = useRef();
  const histogramRef = useRef();
  const correlationRef = useRef();

  // Dataset generation functions
  const generateCircles = (n, noise) => {
    const data = { X: [], y: [] };
    for (let i = 0; i < n; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() < 0.5 ? 0.3 + Math.random() * 0.2 : 0.7 + Math.random() * 0.2;
      const x = radius * Math.cos(angle) + (Math.random() - 0.5) * noise;
      const y = radius * Math.sin(angle) + (Math.random() - 0.5) * noise;
      data.X.push([x, y]);
      data.y.push(radius < 0.5 ? 0 : 1);
    }
    return data;
  };

  const generateMoons = (n, noise) => {
    const data = { X: [], y: [] };
    for (let i = 0; i < n; i++) {
      const t = Math.random() * Math.PI;
      const moonId = Math.random() < 0.5 ? 0 : 1;
      
      let x, y;
      if (moonId === 0) {
        x = Math.cos(t) + (Math.random() - 0.5) * noise;
        y = Math.sin(t) + (Math.random() - 0.5) * noise;
      } else {
        x = 1 - Math.cos(t) + (Math.random() - 0.5) * noise;
        y = 0.5 - Math.sin(t) + (Math.random() - 0.5) * noise;
      }
      
      data.X.push([x, y]);
      data.y.push(moonId);
    }
    return data;
  };

  const generateBlobs = (n, noise) => {
    const data = { X: [], y: [] };
    const centers = [[-1, -1], [1, 1], [-1, 1], [1, -1]];
    
    for (let i = 0; i < n; i++) {
      const clusterId = Math.floor(Math.random() * centers.length);
      const center = centers[clusterId];
      const x = center[0] + (Math.random() - 0.5) * noise * 2;
      const y = center[1] + (Math.random() - 0.5) * noise * 2;
      
      data.X.push([x, y]);
      data.y.push(clusterId % 2);
    }
    return data;
  };

  const generateClassification = (n, noise) => {
    const data = { X: [], y: [] };
    for (let i = 0; i < n; i++) {
      const x1 = (Math.random() - 0.5) * 4;
      const x2 = (Math.random() - 0.5) * 4;
      const x3 = (Math.random() - 0.5) * 2;
      const x4 = (Math.random() - 0.5) * 2;
      
      // Create a complex decision boundary
      const boundary = x1 * x2 + Math.sin(x1) * Math.cos(x2) + x3 * x4;
      const label = boundary > 0 ? 1 : 0;
      
      data.X.push([
        x1 + (Math.random() - 0.5) * noise,
        x2 + (Math.random() - 0.5) * noise,
        x3 + (Math.random() - 0.5) * noise,
        x4 + (Math.random() - 0.5) * noise
      ]);
      data.y.push(label);
    }
    return data;
  };

  // Generate datasets when parameters change
  useEffect(() => {
    const generators = {
      circles: generateCircles,
      moons: generateMoons,
      blobs: generateBlobs,
      classification: generateClassification
    };

    const newDatasets = {};
    Object.keys(generators).forEach(type => {
      newDatasets[type] = generators[type](sampleSize, noiseLevel);
    });

    setDatasets(newDatasets);
    setCurrentDataset(newDatasets[datasetType]);

    if (onParametersChange) {
      onParametersChange({
        datasetType,
        noiseLevel,
        sampleSize,
        data: newDatasets[datasetType]
      });
    }
  }, [datasetType, noiseLevel, sampleSize, onParametersChange]);

  // Calculate dataset statistics
  const statistics = useMemo(() => {
    if (!currentDataset || !currentDataset.X) return null;

    const { X, y } = currentDataset;
    const numFeatures = X[0].length;
    const stats = [];

    for (let i = 0; i < numFeatures; i++) {
      const values = X.map(point => point[i]);
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
      const sorted = [...values].sort((a, b) => a - b);
      
      stats.push({
        name: `Feature ${i + 1}`,
        mean: mean,
        std: Math.sqrt(variance),
        min: Math.min(...values),
        max: Math.max(...values),
        median: sorted[Math.floor(sorted.length / 2)],
        q1: sorted[Math.floor(sorted.length * 0.25)],
        q3: sorted[Math.floor(sorted.length * 0.75)]
      });
    }

    const classDistribution = [0, 0];
    y.forEach(label => classDistribution[label]++);

    return {
      features: stats,
      classes: [
        { label: 'Class 0', count: classDistribution[0], percentage: (classDistribution[0] / y.length) * 100 },
        { label: 'Class 1', count: classDistribution[1], percentage: (classDistribution[1] / y.length) * 100 }
      ],
      total: X.length
    };
  }, [currentDataset]);

  // Render 3D visualization
  useEffect(() => {
    if (!currentDataset || !threejsRef.current || activeTab !== '3d') return;

    const { X, y } = currentDataset;
    const container = threejsRef.current;
    
    // Clear previous content
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // Create points
    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    // Normalize data
    const xExtent = d3.extent(X, d => d[0]);
    const yExtent = d3.extent(X, d => d[1]);
    const xScale = d3.scaleLinear().domain(xExtent).range([-2, 2]);
    const yScale = d3.scaleLinear().domain(yExtent).range([-2, 2]);

    X.forEach((point, i) => {
      const x = xScale(point[0]);
      const y = yScale(point[1]);
      const z = Math.sin(x * 2) * Math.cos(y * 2) * 0.3; // Add 3D effect

      positions.push(x, y, z);

      // Color by class
      const color = new THREE.Color();
      color.setHSL(y[i] === 0 ? 0.6 : 0.0, 0.8, 0.6);
      colors.push(color.r, color.g, color.b);
    });

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      sizeAttenuation: true
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    camera.position.z = 5;

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      points.rotation.y += 0.005;
      renderer.render(scene, camera);
    };
    animate();

    // Cleanup
    return () => {
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [currentDataset, activeTab]);

  // Render 2D scatter plot
  useEffect(() => {
    if (!currentDataset || !scatterRef.current || activeTab !== 'scatter') return;

    const { X, y } = currentDataset;
    const svg = d3.select(scatterRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear()
      .domain(d3.extent(X, d => d[0]))
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(X, d => d[1]))
      .range([height, 0]);

    const colorScale = d3.scaleOrdinal()
      .domain([0, 1])
      .range(['#3b82f6', '#ef4444']);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add points
    g.selectAll(".point")
      .data(X)
      .enter().append("circle")
      .attr("class", "point")
      .attr("cx", d => xScale(d[0]))
      .attr("cy", d => yScale(d[1]))
      .attr("r", 3)
      .attr("fill", (d, i) => colorScale(y[i]))
      .attr("opacity", 0.7)
      .attr("stroke", "white")
      .attr("stroke-width", 0.5);

    // Add labels
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Feature 2");

    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text("Feature 1");
  }, [currentDataset, activeTab]);

  // Render histogram
  useEffect(() => {
    if (!currentDataset || !histogramRef.current || activeTab !== 'stats') return;

    const { X } = currentDataset;
    const svg = d3.select(histogramRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 400 - margin.left - margin.right;
    const height = 300 - margin.top - margin.bottom;

    const g = svg
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const featureValues = X.map(point => point[selectedFeature]);
    const xScale = d3.scaleLinear()
      .domain(d3.extent(featureValues))
      .range([0, width]);

    const histogram = d3.histogram()
      .value(d => d)
      .domain(xScale.domain())
      .thresholds(xScale.ticks(20));

    const bins = histogram(featureValues);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(bins, d => d.length)])
      .range([height, 0]);

    // Add axes
    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(xScale));

    g.append("g")
      .call(d3.axisLeft(yScale));

    // Add bars
    g.selectAll(".bar")
      .data(bins)
      .enter().append("rect")
      .attr("class", "bar")
      .attr("x", d => xScale(d.x0))
      .attr("width", d => Math.max(0, xScale(d.x1) - xScale(d.x0) - 1))
      .attr("y", d => yScale(d.length))
      .attr("height", d => height - yScale(d.length))
      .attr("fill", "#3b82f6")
      .attr("opacity", 0.7);

    // Add labels
    g.append("text")
      .attr("transform", `translate(${width / 2}, ${height + margin.bottom})`)
      .style("text-anchor", "middle")
      .text(`Feature ${selectedFeature + 1} Value`);

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Frequency");
  }, [currentDataset, selectedFeature, activeTab]);

  // Render correlation matrix
  useEffect(() => {
    if (!currentDataset || !correlationRef.current || activeTab !== 'correlation' || currentDataset.X[0].length < 2) return;

    const { X } = currentDataset;
    const numFeatures = X[0].length;
    const svg = d3.select(correlationRef.current);
    svg.selectAll("*").remove();

    const size = 300;
    const cellSize = size / numFeatures;

    // Calculate correlation matrix
    const correlationMatrix = [];
    for (let i = 0; i < numFeatures; i++) {
      correlationMatrix[i] = [];
      for (let j = 0; j < numFeatures; j++) {
        if (i === j) {
          correlationMatrix[i][j] = 1;
        } else {
          const featI = X.map(point => point[i]);
          const featJ = X.map(point => point[j]);
          const meanI = featI.reduce((a, b) => a + b, 0) / featI.length;
          const meanJ = featJ.reduce((a, b) => a + b, 0) / featJ.length;
          
          let numerator = 0, denomI = 0, denomJ = 0;
          for (let k = 0; k < featI.length; k++) {
            const diffI = featI[k] - meanI;
            const diffJ = featJ[k] - meanJ;
            numerator += diffI * diffJ;
            denomI += diffI * diffI;
            denomJ += diffJ * diffJ;
          }
          correlationMatrix[i][j] = numerator / Math.sqrt(denomI * denomJ);
        }
      }
    }

    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
      .domain([-1, 1]);

    // Create heatmap
    for (let i = 0; i < numFeatures; i++) {
      for (let j = 0; j < numFeatures; j++) {
        svg.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize)
          .attr("height", cellSize)
          .attr("fill", colorScale(correlationMatrix[i][j]))
          .attr("stroke", "white")
          .attr("stroke-width", 1);

        svg.append("text")
          .attr("x", j * cellSize + cellSize / 2)
          .attr("y", i * cellSize + cellSize / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .style("font-size", "10px")
          .attr("fill", Math.abs(correlationMatrix[i][j]) > 0.5 ? "white" : "black")
          .text(correlationMatrix[i][j].toFixed(2));
      }
    }

    // Add labels
    for (let i = 0; i < numFeatures; i++) {
      svg.append("text")
        .attr("x", -5)
        .attr("y", i * cellSize + cellSize / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .style("font-size", "12px")
        .text(`F${i + 1}`);

      svg.append("text")
        .attr("x", i * cellSize + cellSize / 2)
        .attr("y", -5)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "end")
        .style("font-size", "12px")
        .text(`F${i + 1}`);
    }
  }, [currentDataset, activeTab]);

  const tabs = [
    { key: '3d', name: '3D Visualization', icon: '🌐' },
    { key: 'scatter', name: '2D Scatter', icon: '📊' },
    { key: 'stats', name: 'Statistics', icon: '📈' },
    { key: 'correlation', name: 'Correlation', icon: '🔗' }
  ];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">🔍</span>
            Enhanced Dataset Preview
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Dataset Controls */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dataset Type
              </label>
              <select
                value={datasetType}
                onChange={(e) => setDatasetType(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="circles">Circles</option>
                <option value="moons">Moons</option>
                <option value="blobs">Blobs</option>
                <option value="classification">Classification</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Noise Level: {noiseLevel.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.05"
                value={noiseLevel}
                onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Sample Size: {sampleSize}
              </label>
              <input
                type="range"
                min="100"
                max="1000"
                step="50"
                value={sampleSize}
                onChange={(e) => setSampleSize(parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <div className="flex items-end">
              <Button
                onClick={() => onDatasetSelect && onDatasetSelect(currentDataset)}
                className="w-full"
              >
                Use Dataset
              </Button>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex space-x-1 mb-6">
            {tabs.map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  activeTab === tab.key
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <span>{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="border rounded-lg p-4 bg-white">
            {activeTab === '3d' && (
              <div>
                <h4 className="text-lg font-semibold mb-4">3D Interactive Visualization</h4>
                <div 
                  ref={threejsRef} 
                  className="w-full h-96 border rounded-lg bg-gray-50"
                  style={{ minHeight: '400px' }}
                />
              </div>
            )}

            {activeTab === 'scatter' && (
              <div>
                <h4 className="text-lg font-semibold mb-4">2D Scatter Plot</h4>
                <div className="flex justify-center">
                  <svg ref={scatterRef}></svg>
                </div>
              </div>
            )}

            {activeTab === 'stats' && (
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold">Feature Statistics</h4>
                  {currentDataset && currentDataset.X[0].length > 1 && (
                    <select
                      value={selectedFeature}
                      onChange={(e) => setSelectedFeature(parseInt(e.target.value))}
                      className="px-3 py-1 border border-gray-300 rounded-md"
                    >
                      {Array.from({ length: currentDataset.X[0].length }, (_, i) => (
                        <option key={i} value={i}>Feature {i + 1}</option>
                      ))}
                    </select>
                  )}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <svg ref={histogramRef}></svg>
                  </div>

                  <div>
                    {statistics && (
                      <div className="space-y-4">
                        <div className="bg-blue-50 p-4 rounded-lg">
                          <h5 className="font-medium text-blue-800 mb-2">Dataset Summary</h5>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <div>Total Samples: <span className="font-bold">{statistics.total}</span></div>
                            <div>Features: <span className="font-bold">{statistics.features.length}</span></div>
                            <div>Class 0: <span className="font-bold">{statistics.classes[0].count}</span></div>
                            <div>Class 1: <span className="font-bold">{statistics.classes[1].count}</span></div>
                          </div>
                        </div>

                        <div className="bg-green-50 p-4 rounded-lg">
                          <h5 className="font-medium text-green-800 mb-2">Feature {selectedFeature + 1} Stats</h5>
                          {statistics.features[selectedFeature] && (
                            <div className="grid grid-cols-2 gap-2 text-sm">
                              <div>Mean: <span className="font-bold">{statistics.features[selectedFeature].mean.toFixed(3)}</span></div>
                              <div>Std Dev: <span className="font-bold">{statistics.features[selectedFeature].std.toFixed(3)}</span></div>
                              <div>Min: <span className="font-bold">{statistics.features[selectedFeature].min.toFixed(3)}</span></div>
                              <div>Max: <span className="font-bold">{statistics.features[selectedFeature].max.toFixed(3)}</span></div>
                              <div>Median: <span className="font-bold">{statistics.features[selectedFeature].median.toFixed(3)}</span></div>
                              <div>IQR: <span className="font-bold">{(statistics.features[selectedFeature].q3 - statistics.features[selectedFeature].q1).toFixed(3)}</span></div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'correlation' && (
              <div>
                <h4 className="text-lg font-semibold mb-4">Feature Correlation Matrix</h4>
                {currentDataset && currentDataset.X[0].length > 1 ? (
                  <div className="flex justify-center">
                    <svg ref={correlationRef} width="350" height="350"></svg>
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    Correlation matrix requires at least 2 features
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Dataset Info */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg">
              <h5 className="font-medium text-blue-800 mb-2">Dataset Type</h5>
              <div className="text-2xl font-bold text-blue-600 capitalize">{datasetType}</div>
              <div className="text-sm text-blue-600">
                {datasetType === 'circles' && 'Concentric circles with noise'}
                {datasetType === 'moons' && 'Two interleaving half circles'}
                {datasetType === 'blobs' && 'Gaussian blobs in clusters'}
                {datasetType === 'classification' && 'Multi-dimensional classification'}
              </div>
            </div>

            <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg">
              <h5 className="font-medium text-green-800 mb-2">Complexity</h5>
              <div className="text-2xl font-bold text-green-600">
                {noiseLevel < 0.1 ? 'Low' : noiseLevel < 0.3 ? 'Medium' : 'High'}
              </div>
              <div className="text-sm text-green-600">Noise level: {(noiseLevel * 100).toFixed(0)}%</div>
            </div>

            <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg">
              <h5 className="font-medium text-purple-800 mb-2">Sample Size</h5>
              <div className="text-2xl font-bold text-purple-600">{sampleSize}</div>
              <div className="text-sm text-purple-600">
                {sampleSize < 300 ? 'Small' : sampleSize < 700 ? 'Medium' : 'Large'} dataset
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default EnhancedDatasetPreview;
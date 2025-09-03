import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

const DatasetVisualizer = ({ 
  data, 
  datasetType, 
  isLoading = false, 
  datasetInfo = null,
  showEnhancedView = false,
  onToggleEnhancedView = null 
}) => {
  const svgRef = useRef(null);
  const [activeView, setActiveView] = useState('scatter');

  useEffect(() => {
    if (!data || isLoading) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 500;
    const height = 350;
    const margin = { top: 40, right: 20, bottom: 60, left: 60 };

    if (activeView === 'scatter') {
      renderScatterPlot(svg, data, width, height, margin, datasetType, datasetInfo);
    } else if (activeView === 'density') {
      renderDensityPlot(svg, data, width, height, margin, datasetType);
    } else if (activeView === 'histogram') {
      renderHistogram(svg, data, width, height, margin, datasetType);
    }

  }, [data, datasetType, isLoading, activeView, datasetInfo]);

  const renderScatterPlot = (svg, data, width, height, margin, datasetType, datasetInfo) => {
    // Set up scales
    const xExtent = d3.extent(data, d => d[0]);
    const yExtent = d3.extent(data, d => d[1]);
    
    const x = d3.scaleLinear()
      .domain([xExtent[0] - 0.5, xExtent[1] + 0.5])
      .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
      .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
      .range([height - margin.bottom, margin.top]);

    const color = d3.scaleOrdinal()
      .domain([0, 1])
      .range(["#3b82f6", "#ef4444"]);

    // Add background grid
    const xTicks = x.ticks(8);
    const yTicks = y.ticks(6);

    svg.append("g")
      .selectAll("line")
      .data(xTicks)
      .join("line")
      .attr("x1", d => x(d))
      .attr("x2", d => x(d))
      .attr("y1", margin.top)
      .attr("y2", height - margin.bottom)
      .attr("stroke", "#e5e7eb")
      .attr("stroke-width", 0.5);

    svg.append("g")
      .selectAll("line")
      .data(yTicks)
      .join("line")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", d => y(d))
      .attr("y2", d => y(d))
      .attr("stroke", "#e5e7eb")
      .attr("stroke-width", 0.5);

    // Add data points with animation
    svg.append("g")
      .selectAll("circle")
      .data(data)
      .join("circle")
      .attr("cx", d => x(d[0]))
      .attr("cy", d => y(d[1]))
      .attr("r", 0)
      .attr("fill", d => color(d[2]))
      .attr("opacity", 0.7)
      .attr("stroke", "white")
      .attr("stroke-width", 1)
      .transition()
      .duration(800)
      .delay((d, i) => i * 2)
      .attr("r", 3.5);

    // Add axes
    svg.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(8))
      .selectAll("text")
      .style("font-size", "12px")
      .style("fill", "#6b7280");

    svg.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(6))
      .selectAll("text")
      .style("font-size", "12px")
      .style("fill", "#6b7280");

    // Add axis labels with feature names if available
    const xLabel = datasetInfo?.feature_names?.[0] || "Feature 1";
    const yLabel = datasetInfo?.feature_names?.[1] || "Feature 2";

    svg.append("text")
      .attr("x", width / 2)
      .attr("y", height - 10)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("fill", "#374151")
      .text(xLabel);

    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("fill", "#374151")
      .text(yLabel);

    // Add title with dataset info
    const title = datasetInfo?.name || `${datasetType.charAt(0).toUpperCase() + datasetType.slice(1)} Dataset`;
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "600")
      .style("fill", "#1f2937")
      .text(title);

    // Add legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 100}, 40)`);

    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 0)
      .attr("r", 6)
      .attr("fill", "#3b82f6");

    legend.append("text")
      .attr("x", 12)
      .attr("y", 5)
      .style("font-size", "12px")
      .style("fill", "#374151")
      .text("Class 0");

    legend.append("circle")
      .attr("cx", 0)
      .attr("cy", 20)
      .attr("r", 6)
      .attr("fill", "#ef4444");

    legend.append("text")
      .attr("x", 12)
      .attr("y", 25)
      .style("font-size", "12px")
      .style("fill", "#374151")
      .text("Class 1");
  };

  const renderDensityPlot = (svg, data, width, height, margin, datasetType) => {
    // Create density contours for each class
    const class0 = data.filter(d => d[2] === 0);
    const class1 = data.filter(d => d[2] === 1);

    const xExtent = d3.extent(data, d => d[0]);
    const yExtent = d3.extent(data, d => d[1]);
    
    const x = d3.scaleLinear()
      .domain([xExtent[0] - 0.5, xExtent[1] + 0.5])
      .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
      .domain([yExtent[0] - 0.5, yExtent[1] + 0.5])
      .range([height - margin.bottom, margin.top]);

    // Add axes
    svg.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(8));

    svg.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(6));

    // Add data points
    svg.append("g")
      .selectAll("circle")
      .data(data)
      .join("circle")
      .attr("cx", d => x(d[0]))
      .attr("cy", d => y(d[1]))
      .attr("r", 2)
      .attr("fill", d => d[2] === 0 ? "#3b82f6" : "#ef4444")
      .attr("opacity", 0.6);

    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "600")
      .text("Density View");
  };

  const renderHistogram = (svg, data, width, height, margin, datasetType) => {
    // Create histograms for each feature
    const feature1Data = data.map(d => d[0]);
    const feature2Data = data.map(d => d[1]);

    const x = d3.scaleLinear()
      .domain(d3.extent(feature1Data))
      .range([margin.left, width - margin.right]);

    const histogram = d3.histogram()
      .domain(x.domain())
      .thresholds(x.ticks(20));

    const bins = histogram(feature1Data);

    const y = d3.scaleLinear()
      .domain([0, d3.max(bins, d => d.length)])
      .range([height - margin.bottom, margin.top]);

    // Add bars
    svg.append("g")
      .selectAll("rect")
      .data(bins)
      .join("rect")
      .attr("x", d => x(d.x0))
      .attr("y", d => y(d.length))
      .attr("width", d => x(d.x1) - x(d.x0) - 1)
      .attr("height", d => y(0) - y(d.length))
      .attr("fill", "#3b82f6")
      .attr("opacity", 0.7);

    // Add axes
    svg.append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x));

    svg.append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y));

    // Add title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", 20)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "600")
      .text("Feature 1 Distribution");
  };

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="flex items-center justify-center h-80">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600">Generating dataset...</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Dataset Visualization</CardTitle>
          <div className="flex items-center space-x-2">
            {/* View Toggle Buttons */}
            <div className="flex rounded-lg border border-gray-200 p-1">
              <button
                onClick={() => setActiveView('scatter')}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  activeView === 'scatter'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                Scatter
              </button>
              <button
                onClick={() => setActiveView('density')}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  activeView === 'density'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                Density
              </button>
              <button
                onClick={() => setActiveView('histogram')}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  activeView === 'histogram'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                Histogram
              </button>
            </div>
            
            {/* Enhanced View Toggle */}
            {onToggleEnhancedView && (
              <Button
                variant={showEnhancedView ? "default" : "outline"}
                size="sm"
                onClick={onToggleEnhancedView}
              >
                {showEnhancedView ? "Simple View" : "Enhanced View"}
              </Button>
            )}
          </div>
        </div>
        
        {/* Dataset Info */}
        {datasetInfo && (
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <Badge variant="secondary">
              {datasetInfo.n_samples} samples
            </Badge>
            <Badge variant="secondary">
              {datasetInfo.n_features} features
            </Badge>
            {datasetInfo.metadata?.dataset_complexity && (
              <Badge variant="outline">
                Complexity: {(datasetInfo.metadata.dataset_complexity.overall * 100).toFixed(0)}%
              </Badge>
            )}
          </div>
        )}
      </CardHeader>
      
      <CardContent className="pt-0">
        <div className="bg-white rounded-lg border">
          <svg 
            ref={svgRef} 
            width="100%" 
            height="350"
            viewBox="0 0 500 350"
            className="max-w-full h-auto"
          />
        </div>
        
        {/* Dataset Description */}
        {datasetInfo?.description && (
          <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700">{datasetInfo.description}</p>
          </div>
        )}
        
        {/* Feature Information */}
        {datasetInfo?.feature_names && datasetInfo.feature_names.length > 2 && (
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">All Features:</h4>
            <div className="flex flex-wrap gap-1">
              {datasetInfo.feature_names.map((name, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {name}
                </Badge>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-2">
              Showing first 2 features in visualization. Use Enhanced View for complete analysis.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default DatasetVisualizer;